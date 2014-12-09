#include "edge_parallel.cuh"
#include "common.cuh"

std::vector< std::vector<int> > multi_search_edge_parallel_setup(const device_graph &g, int start, int end)
{
	//For now, use "standard" grid/block sizes. These can be tuned later on.
	dim3 dimGrid, dimBlock;
        //Returns number of source vertices to store for verification purposes
        size_t sources_to_store = configure_grid(dimGrid,dimBlock,start,end);

	//Device pointers
	int *d_d;
	size_t pitch_d;
	cudaEvent_t start_event, end_event;

	//Allocate algorithm-specific memory
	start_clock(start_event,end_event);
	checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch_d,sizeof(int)*g.n,sources_to_store));
	size_t GPU_memory_requirement = sizeof(int)*g.n*sources_to_store + 2*sizeof(int)*(g.m);
	std::cout << "Edge parallel memory requirement: " << GPU_memory_requirement/(1 << 20) << " MB" << std::endl;

	multi_search_edge_parallel<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.F.data()),thrust::raw_pointer_cast(g.C.data()),g.n,g.m,d_d,pitch_d,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        std::vector< std::vector<int> > d_host_vector;
        transfer_result(g,d_d,pitch_d,sources_to_store,d_host_vector);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for baseline edge-parallel algorithm: " << time << " s" << std::endl;

	return d_host_vector;
}

__global__ void multi_search_edge_parallel(const int *F, const int *C, const int n, const int m, int *d, size_t pitch_d, const int start, const int end)
{
	int j = threadIdx.x;

	for(int i=blockIdx.x+start; i<end; i+=gridDim.x)
	{
		//Initialization
		int *d_row = (int *)((char *)d + blockIdx.x*pitch_d); 
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

		
		__shared__ bool done;
		__shared__ int current_depth;
		if(j == 0)
		{
			done = false;
			current_depth = -1;
		}
		__syncthreads();

		while(!done)
		{
			__syncthreads(); //This barrier is necessary in the event that thread 0 sets done to true before other threads check the while loop
			if(j == 0)
			{
				done = true;
				current_depth++;
			}
			__syncthreads();
			for(int k=threadIdx.x; k<m; k+=blockDim.x) //m undirected edges = 2m directed edges. 
			{
				int v = F[k];
				if(d_row[v] == current_depth)
				{
					int w = C[k];
					if(atomicCAS(&d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
					{
						done = false;
					}
				}
			}
			__syncthreads();
		}
	}
}
