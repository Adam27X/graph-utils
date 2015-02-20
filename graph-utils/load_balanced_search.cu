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

//Speciailized binary search for the LBS problem: We want to return the greatest index in array that is less than key
__device__ int binary_search(int *array, int array_size, int key, int low = 0)
{
	//int low = 0;
	int high = array_size-1;

	while(low <= high)
	{
		int mid = low + ((high - low) / 2);
		int midVal = array[mid];

		if(midVal < key)
		{
			low = mid + 1;
		}
		else if(midVal > key)
		{
			high = mid - 1;
		}
		else
		{
			/*while(mid < high)
			{
				if(array[mid+1] == array[mid])
				{
					mid++;
				}
				else
				{
					break;
				}
			}*/
			return mid; //guarantee O(log n), and use the "normal method" to ensure the right answer is found
		}
	}

	return high; //key not found - return the lower key since we want the greatest index *less* than the key
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
	if(vertex_frontier_rounded < WARP_SIZE)
	{
		vertex_frontier_rounded = WARP_SIZE; //Must be at least the size of the warp for the syncthreads in the next loop to work correctly
	}
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
		while(ind < vertex_frontier_size && i >= scanned_edges[ind])
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

	//This is actually way slower than the naive approach, at least for the inputs I've tested so far. Perhaps that input isn't large enough?
	/*for(int i=getLaneId(); i<total_edges; i+=WARP_SIZE)
        {
		if(i != getLaneId())
		{
			result[i] = binary_search(scanned_edges,vertex_frontier_size,i,result[i-WARP_SIZE]);
		}
		else
		{
			result[i] = binary_search(scanned_edges,vertex_frontier_size,i);	
		}
        }*/
}

//TODO: Reorganize so that each thread has multiple items to scan at once (check occupancy for this), tuning
__device__ void load_balance_search_block(const int vertex_frontier_size, int *edge_frontier_size, const int *edge_counts, int *scanned_edges, int *result)
{
	typedef cub::BlockScan<int,BLOCK_SIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;

	int total_edges = 0;
	//Ensure all threads in the block execute BlockScan and get the value of current_edges
	__shared__ int vertex_frontier_rounded;
	if(threadIdx.x == 0)
	{	
		vertex_frontier_rounded = get_next_power_of_2(vertex_frontier_size);
		if(vertex_frontier_rounded < blockDim.x)
		{
			vertex_frontier_rounded = blockDim.x; //Must be at least the size of the warp for the syncthreads in the next loop to work correctly
		}
	}
	__syncthreads();
	for(int i=threadIdx.x; i<vertex_frontier_rounded; i+=blockDim.x) 
	{
		int local_count[ITEMS_PER_THREAD];
		for(int j=0; j<ITEMS_PER_THREAD; j++)
		{
			if((ITEMS_PER_THREAD*i)+j < vertex_frontier_size)
			{
				local_count[j] = edge_counts[ITEMS_PER_THREAD*i+j];
			}
			else
			{
				local_count[j] = 0;
			}
		}

		int current_edges;
		BlockScan(temp_storage).ExclusiveSum(local_count,local_count,current_edges);
		__syncthreads(); //Needed for reuse of WarpScan
		
		for(int j=0; j<ITEMS_PER_THREAD; j++)
		{
			if((ITEMS_PER_THREAD*i)+j < vertex_frontier_size)
			{
				scanned_edges[ITEMS_PER_THREAD*i+j] = local_count[j] + total_edges;
			}
		}
		total_edges += current_edges;
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		edge_frontier_size[0] = scanned_edges[vertex_frontier_size-1]+edge_counts[vertex_frontier_size-1];
	}
	__syncthreads();

	//LBS work below takes multiple orders of magnitude longer than scanning work above
	int ind = 0;
	for(int i=threadIdx.x; i<edge_frontier_size[0]; i+=blockDim.x) 
	{
		while(ind < vertex_frontier_size && i >= scanned_edges[ind]) 
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

__global__ void extract_edges_warp(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges)
{
	load_balance_search_warp(vertex_frontier_size,edges,edge_counts,scanned_edges,result);
}

__global__ void extract_edges_block(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges)
{
	load_balance_search_block(vertex_frontier_size,edges,edge_counts,scanned_edges,result);
}
