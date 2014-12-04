#include "edge_parallel.cuh"

std::vector< std::vector<int> > multi_search_edge_parallel_setup(device_graph &g, int start, int end)
{
	//For now, use "standard" grid/block sizes. These can be tuned later on.
	dim3 dimGrid, dimBlock;
	dimGrid.x = 14;
	dimGrid.y = 1;
	dimGrid.z = 1;

	dimBlock.x = 1024;
	dimBlock.y = 1;
	dimBlock.z = 1;

	//Device pointers
	int *d_d;
	size_t pitch_d;
	cudaEvent_t start_event, end_event;

	//Number of sources to store
	size_t sources_to_store;
	if((end-start) > 14)
	{
		sources_to_store = end-start;
	}
	else
	{
		sources_to_store = 14;
	}	

	//Allocate algorithm-specific memory
	start_clock(start_event,end_event);
	checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch_d,sizeof(int)*g.n,sources_to_store));

	multi_search_edge_parallel<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.F.data()),thrust::raw_pointer_cast(g.C.data()),g.n,g.m,d_d,pitch_d,start,end);
	checkCudaErrors(cudaPeekAtLastError());

	//Transfer result to host. Use CUDA library calls to copy into a C-style array and then move that to a vector for convenience.
	int *d_host_array = new int[g.n*sources_to_store];
	checkCudaErrors(cudaMemcpy2D(d_host_array,sizeof(int)*g.n,d_d,pitch_d,sizeof(int)*g.n,sources_to_store,cudaMemcpyDeviceToHost));
	std::vector< std::vector<int> > d_host_vector(sources_to_store);
	for(int i=0; i<sources_to_store; i++)
	{
		d_host_vector[i].resize(g.n);
		for(int j=0; j<g.n; j++)
		{
			d_host_vector[i][j] = d_host_array[i*g.n + j];
		}
	}
	delete[] d_host_array;

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
			if(j == 0)
			{
				done = true;
				current_depth++;
			}
			__syncthreads();
			for(int k=threadIdx.x; k<m; k+=blockDim.x) //m undirected edges = 2m directed edges. Might want to change this nomenclature.
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
