#pragma once

#include <vector>
#include <cuda.h>
#include <cstdio>
#include <iostream>

#include "../../parse.h"
#include "../../device_graph.h"
#include "../../util_device.cuh"
#include "../race_and_resolve.cuh"

// How to adjust this algorithm to easily extend to the following problems:
// Diameter sampling (needs a global variable for keeping track of the maximum distance seen from each source)
// Betweenness Centrality (one call for BFS, another call for dependency accum? Hard to store all of the intermediate information...some sort of template-based if statement might be better)
// All-pairs shortest path
// Reachability Querying

// Can Lambda Expressions be leveraged here? Yes. The core function needs to be a __device__ function and then its uses can all be global functions (the multi_search global function will be mostly empty)
__device__ void multi_search(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end)
{
        int j = threadIdx.x;
        int lane_id = getLaneId();
        __shared__ int  *Q_row;
        __shared__ int *Q2_row;

        if(j == 0)
        {
                Q_row = (int*)((char*)Q + blockIdx.x*pitch_Q);
                Q2_row = (int*)((char*)Q2 + blockIdx.x*pitch_Q2);
        }
        __syncthreads();

        for(int i=blockIdx.x+start; i<end; i+=gridDim.x)
        {
                int *d_row = (int*)((char*)d + blockIdx.x*pitch_d);
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
                }
                __syncthreads();

                __shared__ int Q_len;
                __shared__ int Q2_len;

                if(j == 0)
                {
                        Q_row[0] = i;
                        Q_len = 1;
                        Q2_len = 0;
                }
                __syncthreads();

                while(1)
                {
                        //Listing 9: Warp-based, strip-mined neighbor gathering
                        int v, r, r_end;
                        int k = threadIdx.x;

                        if(k < Q_len)
                        {
                                v = Q_row[k];
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
                                                if(d_row[w] == INT_MAX)
                                                {
                                                        d_row[w] = d_row[v_new]+1;
                                                        int t = atomicAdd(&Q2_len,1);
                                                        Q2_row[t] = w;
                                                }
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
                                        r = R[v];
                                        r_end = R[v+1];
                                }
                                else
                                {
                                        v = -1;
                                        r = 0;
                                        r_end = 0;
                                }

                                //if((lane_id == 0) && (k >= Q_len)) //The entire warp is done
                                if((k-threadIdx.x) >= Q_len) //If thread 0 doesn't have work, the entire warp is done
                                {
                                        break;
                                }
                        }
                        __syncthreads();
                                        
                        if(Q2_len == 0)
                        {
                                break;
				//TODO: Call functor to search d_row for it's max value, reachability?
                        }
                        else
                        {
                                for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
                                {
                                        Q_row[kk] = Q2_row[kk];
                                }
                                __syncthreads();

                                if(j==0)
                                {
                                        Q_len = Q2_len;
                                        Q2_len = 0;
                                }
                                __syncthreads();
                        }
                }
        }
}

__global__ void multi_search_shuffle_based(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end)
{
	multi_search(R,C,n,d,pitch_d,Q,pitch_Q,Q2,pitch_Q2,start,end);	
}

