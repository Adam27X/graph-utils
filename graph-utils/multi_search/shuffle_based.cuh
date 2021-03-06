#pragma once

#include <vector>
#include <cuda.h>
#include <cstdio>
#include <iostream>

#include <cub/cub.cuh>

#include "../../parse.h"
#include "../../device_graph.h"
#include "../../util_device.cuh"
#include "../race_and_resolve.cuh"
#include "../load_balanced_search.cuh"

std::vector< std::vector<int> > multi_search_shuffle_based_setup(const device_graph &g, const program_options &op, int start, int end);

// How to adjust this algorithm to easily extend to the following problems:
// Diameter sampling (needs a global variable for keeping track of the maximum distance seen from each source)
// Betweenness Centrality (one call for BFS, another call for dependency accum? Hard to store all of the intermediate information...some sort of template-based if statement might be better)
// All-pairs shortest path
// Reachability Querying

struct pitch
{
	pitch() : d(0),sigma(0),delta(0),bc(0),Q(0),Q2(0),S(0),endpoints(0),edge_counts(0),scanned_edges(0),LBS(0) { }

	size_t d;
	size_t sigma;
	size_t delta;
	size_t bc;
	size_t Q;
	size_t Q2;
	size_t S;
	size_t endpoints;
	size_t edge_counts;
	size_t scanned_edges;
	size_t LBS;
	size_t Q_small;
	size_t Q2_small;
};

//Get block of data from pitched pointer and pitch size
template <typename T>
__device__ __forceinline__ T* get_row(T* data, size_t p)
{
	return (T*)((char*)data + blockIdx.x*p);	
}

//return vertex outdegree
__device__ __forceinline__ int outdegree(int v, const int *R)
{
	        return __ldg(&R[v+1])-__ldg(&R[v]);
}

//TODO: Check for bugs in here. Can replace with my sandbox if necessary. It appears we can't use this in both clauses of an if/else
// warp-aggregated atomic increment
__device__ __forceinline__ int atomicAggInc(int *ctr)
{
        // mask of active lanes
        int mask = __ballot(1);

        // select the leader
        int leader = __ffs(mask) - 1;

        // leader does the update
        int res;
        if(getLaneId() == leader)
        res = atomicAdd(ctr, __popc(mask));

        // broadcast result
        res = __shfl(res, leader);

        // each thread computes its own value
        return res + __popc(mask & ((1 << getLaneId()) - 1));
}

#define Q_THRESHOLD 0

//TODO: Generalize the beginning and endroutines for more flexibility. Push initialization into lambdas and, if necessary, reuse the lambdas. Use traits and provide default (likely null) values for some of the parameters
template <class F1, class F2, class F3, class F4, class F5, class F6, class F7>
__device__ void multi_search(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, int *Q_small, int *Q2_small, const pitch p, const int start, const int end, F1 getMax, F2 initLocal, F3 visitVertex, F4 initStack, F5 insertStack, F6 updateEndpoints, F7 dependencyAccum)
{
        int j = threadIdx.x;
        int lane_id = getLaneId();
	int warp_id = threadIdx.x/32;
        __shared__ int  *Q_row;
        __shared__ int *Q2_row;

	__shared__ int *Q_small_row;
	__shared__ int *Q2_small_row;

        if(j == 0)
        {
		Q_row = get_row(Q,p.Q);
		Q2_row = get_row(Q2,p.Q2);

		Q_small_row = get_row(Q_small,p.Q_small);
		Q2_small_row = get_row(Q2_small,p.Q2_small);
        }
        __syncthreads();

	//TODO: Push up declarations of variables and combine initLocal and initStack
        for(int i=blockIdx.x+start; i<end; i+=gridDim.x)
        {
		int *d_row = get_row(d,p.d);
                for(int k=threadIdx.x; k<n; k+=blockDim.x)
                {
                        if(k == i)
                        {
                                d_row[k] = 0;
                        }
                        else
                        {
                                d_row[k] = INT_MAX;
                        }
			initLocal(k,i);
                }
                __syncthreads();

                __shared__ int Q_len;
                __shared__ int Q2_len;
		__shared__ int Q_small_len;
		__shared__ int Q2_small_len;
		__shared__ bool degree_zero_source;

                if(j == 0)
                {
			if(outdegree(i,R) < Q_THRESHOLD)
			{
				Q_small_row[0] = i;
				Q_small_len = 1;
				Q_len = 0;
			}
			else
			{
				Q_row[0] = i;
				Q_len = 1;
				Q_small_len = 0;
			}
                        Q2_len = 0;
			Q2_small_len = 0;
			initStack(i);

			if(__ldg(&R[i+1]) == __ldg(&R[i])) //human-readable: if(outdegree == 0)
			{
				degree_zero_source = true;
			}
			else
			{
				degree_zero_source = false;
			}
                }
                __syncthreads();

		//Don't waste time traversing vertices of degree zero
		if(degree_zero_source)
		{
			continue;
		}


                while(1) //While a frontier exists for this source vertex...
                {
			/*//Handle the high-degree vertices first, using the entire block
			for(int z=0; z<Q_small_len; z++)
			{
				int v = Q_small_row[z];
				for(int r=__ldg(&R[v])+threadIdx.x; r<__ldg(&R[v+1]); r+=blockDim.x)
				{
					int w = __ldg(&C[r]);
					if(d_row[w] == INT_MAX)
					{
						d_row[w] = d_row[v] + 1;
						if(outdegree(w,R) > Q_THRESHOLD)
						{
							//int t = atomicAggInc(&Q2_small_len); //Use atomicAggInc here for maximum throughput
							int t = atomicAdd(&Q2_small_len,1);
							Q2_small_row[t] = w;
						}
						else
						{
							int t = atomicAggInc(&Q2_len);
							//int t = atomicAdd(&Q2_len,1);
							Q2_row[t] = w;
						}
					}
				}
			}
			__syncthreads();*/

			//Let each warp be assigned to an element in the queue, once that element is processed the warp grabs the next queue element, if any. Warps synchronize once all queue elements are handled.
			__shared__ int next_queue_element;
			int k = warp_id; //current_queue_element
			
			//number of warps. Warps [0,w-1] process queue elements [0,w-1] in the current frontier and asynchronously grab elements [w,Q_len). 
			if(j == 0)
			{	
				next_queue_element = blockDim.x/WARP_SIZE; 
			}
			__syncthreads();

			while(k < Q_len) //Some warps will execute this loop, some won't. When a warp does, all threads in the warp do.
			{
				int v = Q_row[k];
				int r = __ldg(&R[v]) + lane_id;
				int r_end = __ldg(&R[v+1]);

				while(r < r_end) //Only some threads in each warp will execute this loop
				{
					int w = __ldg(&C[r]);

					//atomics are only needed here when we're computing shortest path calculations
					visitVertex(d_row,v,w,Q2_row,&Q2_len,Q2_small_row,&Q2_small_len);

					r += WARP_SIZE;
				}	
				if(lane_id == 0)
				{
					k = atomicAdd(&next_queue_element,1); //Grab the next item off of the queue
				}
				k = __shfl(k,0); //All threads in the warp need the value of k
			}
			__syncthreads();

			//Separately handle vertices with especially small vertex frontiers
			for(int z=threadIdx.x; z<Q_small_len; z+=blockDim.x)
			{
				int v = Q_small_row[z];
				for(int r=__ldg(&R[v]); r<__ldg(&R[v+1]); r++)
				{
					int w = __ldg(&C[r]);
					visitVertex(d_row,v,w,Q2_row,&Q2_len,Q2_small_row,&Q2_small_len);
				}
			}
			__syncthreads();

		        //TODO: Combine getMax, insertStack, and updateEndpoints into one functon that resets the queue	
                        if(Q2_len == 0 && Q2_small_len == 0)
                        {
				for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
				{
					getMax(d_row[kk]);
				}
                                break;
                        }
                        else
                        {
				//Try swapping pointers instead
				for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
				{
					insertStack(Q2_row,kk);
				}
				for(int kk=threadIdx.x; kk<Q2_small_len; kk+=blockDim.x)
				{
					insertStack(Q2_small_row,kk); //Use a separate stack for high-degree vertices as well?
				}
				__syncthreads();
				if(j == 0)
				{
					int *tmp = Q_row;
					Q_row = Q2_row;
					Q2_row = tmp;

					tmp = Q_small_row;
					Q_small_row = Q2_small_row;
					Q2_small_row = tmp;
				}
                                __syncthreads();

                                if(j==0)
                                {
					updateEndpoints(Q2_len);
                                        Q_len = Q2_len;
                                        Q2_len = 0;
					Q_small_len = Q2_small_len;
					Q2_small_len = 0;
                                }
                                __syncthreads();
                        }
                }

		__syncthreads();
		dependencyAccum(d_row,i,j,lane_id);
        }
}

__global__ void multi_search_shuffle_based(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, int *Q_small, int *Q2_small, const pitch p, const int start, const int end);
/*__global__ void diameter_sampling(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, int *max, const pitch p, const int start, const int end);
__global__ void count_shortest_paths(const int *R, const int *C, const int n, int *d, unsigned long long *sigma, int *Q, int *Q2, const pitch p, const int start, const int end);
__global__ void betweenness_centrality(const int *R, const int *C, const int *F, const int n, const int m, int *d, unsigned long long *sigma, float *delta, float *bc, int *Q, int *Q2, int *S, int *endpoints, const pitch p, const int start, const int end, int *dep_accum);
__global__ void transitive_closure(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, const pitch p, const int start, const int end);*/
