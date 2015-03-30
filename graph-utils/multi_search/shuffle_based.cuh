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

std::vector< std::vector<int> > multi_search_shuffle_based_setup(const device_graph &g, int start, int end);

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
};

//Get block of data from pitched pointer and pitch size
template <typename T>
__device__ __forceinline__ T* get_row(T* data, size_t p)
{
	return (T*)((char*)data + blockIdx.x*p);	
}

//TODO: Generalize the beginning and endroutines for more flexibility. Push initialization into lambdas and, if necessary, reuse the lambdas
template <class F1, class F2, class F3, class F4, class F5, class F6, class F7>
__device__ void multi_search(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, const pitch p, const int start, const int end, F1 getMax, F2 initLocal, F3 updateSigma, F4 initStack, F5 insertStack, F6 updateEndpoints, F7 dependencyAccum)
{
        int j = threadIdx.x;
        int lane_id = getLaneId();
	int warp_id = threadIdx.x/32;
        __shared__ int  *Q_row;
        __shared__ int *Q2_row;

        if(j == 0)
        {
		Q_row = get_row(Q,p.Q);
		Q2_row = get_row(Q2,p.Q2);
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

                if(j == 0)
                {
                        Q_row[0] = i;
                        Q_len = 1;
                        Q2_len = 0;
			initStack(i);
                }
                __syncthreads();

                while(1)
                {
                        //Listing 9: Warp-based, strip-mined neighbor gathering
                        int v, r, r_end;
                        int k = lane_id*WARP_SIZE + warp_id;

                        if(k < Q_len)
                        {
                                v = Q_row[k];
				//v = Q_row[lane_id*WARP_SIZE + warp_id]; //strided access into the queue to provide better load balancing across warps
                                r = R[v];
                                r_end = R[v+1];
                        }
                        else
                        {
                                v = -1;
                                r = 0;
                                r_end = 0;
                        }

                        while(1)
                        {
                                while(__any(r_end-r))
                                {
                                        //Vie for control of warp
                                        int winner = race_and_resolve_warp(r_end-r);

                                        //Strip mine winner's adjlist
                                        int r_gather = __shfl(r,winner) + lane_id; 
                                        int r_gather_end = __shfl(r_end,winner); 
                                        int v_new = __shfl(v,winner);
                                        while(r_gather < r_gather_end)
                                        {
                                                int w = C[r_gather];

                                                //Assuming no duplicate/self-edges in the graph, no atomics needed
                                                //if(d_row[w] == INT_MAX)
						//atomicCAS is necessary here for appropriately computing the number of shortest paths
						//this restriction can be lifted if the calculation of d is all that matters. 
						//This discrepancy can be handled via lambdas, but is of low priority as of right now.
						//TODO: Make this entire section its own function call so atomics are only used where necessary
						if(atomicCAS(&d_row[w],INT_MAX,d_row[v_new]+1) == INT_MAX)
                                                {
                                                        int t = atomicAdd(&Q2_len,1);
                                                        Q2_row[t] = w;
                                                }
						updateSigma(d_row,v_new,w);

                                                r_gather += WARP_SIZE;
                                        }

                                        if(winner == lane_id) //Same thread cannot win twice
                                        {
                                                r = 0;
                                                r_end = 0;
                                        }
                                }

                                k+=blockDim.x;
                                if(k < Q_len)
                                {
					v = Q_row[k];
					//v = Q_row[lane_id*WARP_SIZE + warp_id]; //strided access into the queue to provide better load balancing across warps
                                        r = R[v];
                                        r_end = R[v+1];
                                }
                                else
                                {
                                        v = -1;
                                        r = 0;
                                        r_end = 0;
                                }

                                if((k-(lane_id*WARP_SIZE)) >= Q_len) //If thread 0 doesn't have work, the entire warp is done
                                {
                                        break;
                                }
                        }
                        __syncthreads();
                                       
		        //TODO: Combine getMax, insertStack, and updateEndpoints into one functon that resets the queue	
                        if(Q2_len == 0)
                        {
				for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
				{
					getMax(d_row[kk]);
				}
                                break;
                        }
                        else
                        {
                                for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
                                {
                                        Q_row[kk] = Q2_row[kk];
					insertStack(Q2_row,kk);
                                }
                                __syncthreads();

                                if(j==0)
                                {
					updateEndpoints(Q2_len);
                                        Q_len = Q2_len;
                                        Q2_len = 0;
                                }
                                __syncthreads();
                        }
                }

		__syncthreads();
		dependencyAccum(d_row,i,j,lane_id);
        }
}

__global__ void multi_search_shuffle_based(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, const pitch p, const int start, const int end);
__global__ void diameter_sampling(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, int *max, const pitch p, const int start, const int end);
__global__ void count_shortest_paths(const int *R, const int *C, const int n, int *d, unsigned long long *sigma, int *Q, int *Q2, const pitch p, const int start, const int end);
__global__ void betweenness_centrality(const int *R, const int *C, const int *F, const int n, const int m, int *d, unsigned long long *sigma, float *delta, float *bc, int *Q, int *Q2, int *S, int *endpoints, const pitch p, const int start, const int end, int *dep_accum);
__global__ void transitive_closure(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, const pitch p, const int start, const int end);
