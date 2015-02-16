#include "load_balanced_search.cuh"

//TODO: Scale to larger sets of edges, extend to blocks, etc.
//More efficient for threads to process consecutive elements in results rather than stride-32 elements?

//Use a warp to extract edges that need to be traversed
//Does the edge frontier size need to be stored to global memory? It can be obtained directly from the scan and edge counts? If it doesn't affect performance much then it's worth keeping I guess.
__device__ void load_balance_search_warp(int vertex_frontier_size, int *edge_frontier_size, int *edge_counts, int *scanned_edges, int *result)
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
	if(getLaneId() == 0)
	{
		edge_frontier_size[0] = scanned_edges[vertex_frontier_size-1]+edge_counts[vertex_frontier_size-1];
	}
	int total_edges = __shfl(edge_frontier_size[0],0);

        int ind = 0;
        int i = threadIdx.x;
        while(i < total_edges)
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

__global__ void extract_edges(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges)
{
	load_balance_search_warp(vertex_frontier_size,edges,edge_counts,scanned_edges,result);
}

