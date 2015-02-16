#include "load_balanced_search.cuh"

//TODO: Scale to larger sets of edges, make scanning and LBS itself it's own callable device function, extend to blocks, etc.
//More efficient for threads to process consecutive elements in results rather than stride-32 elements?
__device__ void load_balance_search_warp(int num_edges, int *scanned_edges, int *result)
{
        int ind = 0;
        int i = threadIdx.x;
        while(i < num_edges)
        {
                //better: binary search this array
                while(i >= scanned_edges[ind])
                {
                        ind++;
                }
                result[i] = ind-1;
                i += WARP_SIZE;
        }
}

//Use a warp to extract edges that need to be traversed
__global__ void extract_edges(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges)
{
        __shared__ typename cub::WarpScan<int>::TempStorage temp_storage;
        int local_count;
        if(threadIdx.x < vertex_frontier_size)
        {
                local_count = edge_counts[threadIdx.x];
        }
        else
        {
                local_count = 0;
        }

        cub::WarpScan<int>(temp_storage).ExclusiveSum(local_count,scanned_edges[threadIdx.x]);
        if(threadIdx.x == 0)
        {
                edges[0] = scanned_edges[vertex_frontier_size-1]+edge_counts[vertex_frontier_size-1];
        }
        int total_edges = __shfl(edges[0],0);
        load_balance_search_warp(total_edges,scanned_edges,result);
}

