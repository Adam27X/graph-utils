#include "load_balanced_search.cuh"

__device__ int __forceinline__ get_next_power_of_2(int x)
{
	if(x < 0)
	{
		return 0;
	}
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x>> 16;

	return x+1;
}

//TODO: Scale to larger sets of edges, extend to blocks, etc. 
//More efficient for threads to process consecutive elements in results rather than stride-32 elements?

//Use a warp to extract edges that need to be traversed
//Does the edge frontier size need to be stored to global memory? It can be obtained directly from the scan and edge counts? If it doesn't affect performance much then it's worth keeping I guess.
__device__ void load_balance_search_warp(const int vertex_frontier_size, int *edge_frontier_size, const int *edge_counts, int *scanned_edges, int *result)
{
	__shared__ typename cub::WarpScan<int>::TempStorage temp_storage;

	int total_edges = 0;
	//Ensure all threads in the warp execute WarpScan and get the value of total_edges
	int vertex_frontier_rounded = get_next_power_of_2(vertex_frontier_size);
	for(int i=getLaneId(); i<vertex_frontier_rounded; i+=WARP_SIZE) 
	{
		int local_count = i < vertex_frontier_size ? edge_counts[i] : 0;
		int current_edges;
		cub::WarpScan<int>(temp_storage).ExclusiveSum(local_count,scanned_edges[i],current_edges);
		__syncthreads(); //Needed for reuse of WarpScan
		if((i != getLaneId()) && (i < vertex_frontier_size))
		{
			scanned_edges[i] += total_edges; //Add previous number of edges for subsequent loop iterations
		}
		total_edges += current_edges;
	}
	if(getLaneId() == 0)
	{
		edge_frontier_size[0] = scanned_edges[vertex_frontier_size-1]+edge_counts[vertex_frontier_size-1];
	}

        int ind = 0;
	for(int i=getLaneId(); i<total_edges; i+=WARP_SIZE)
        {
                //better: binary search this array
                while(i >= scanned_edges[ind])
                {
                        ind++;
                }
		if(ind >= vertex_frontier_size) //boundary condition
		{
			result[i] = vertex_frontier_size-1;
		}
		else
		{
                	result[i] = ind-1;
		}
        }
}

__global__ void extract_edges(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges)
{
	load_balance_search_warp(vertex_frontier_size,edges,edge_counts,scanned_edges,result);
}

