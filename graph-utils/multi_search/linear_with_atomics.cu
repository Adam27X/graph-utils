#include "linear_with_atomics.cuh"
#include "common.cuh"

//TODO: Device graph can be passed as const here?
std::vector< std::vector<int> > multi_search_linear_atomics_setup(device_graph &g, int start, int end)
{
	dim3 dimGrid, dimBlock;
	//Returns number of source vertices to store for verification purposes
	size_t sources_to_store = configure_grid(dimGrid,dimBlock,start,end); 

	//Device pointers
	int *d_d, *Q_d, *Q2_d;
	size_t pitch_d, pitch_Q, pitch_Q2;
	cudaEvent_t start_event, end_event;

	//Allocate algorithm-specific memory
	start_clock(start_event,end_event);
	checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch_d,sizeof(int)*g.n,sources_to_store));
	checkCudaErrors(cudaMallocPitch((void**)&Q_d,&pitch_Q,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&pitch_Q2,sizeof(int)*g.n,dimGrid.x));

	multi_search_linear_atomics<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),g.n,d_d,pitch_d,Q_d,pitch_Q,Q2_d,pitch_Q2,start,end);
	checkCudaErrors(cudaPeekAtLastError());

	std::vector< std::vector<int> > d_host_vector;
	transfer_result(g,d_d,pitch_d,sources_to_store,d_host_vector);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for baseline linear algorithm with atomics: " << time << " s" << std::endl;

	return d_host_vector;

}

__global__ void multi_search_linear_atomics(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end)
{
	int j = threadIdx.x;
	__shared__ int *Q_row;
	__shared__ int *Q2_row;

	if(j == 0)
	{
		Q_row = (int *)((char*)Q + blockIdx.x*pitch_Q);
		Q2_row = (int *)((char*)Q2 + blockIdx.x*pitch_Q2);
	}
	__syncthreads();

	//Outer loop over all source vertices
	for(int i=blockIdx.x+start; i<end; i+=gridDim.x)
	{
		//Initialization
		int *d_row = (int *)((char*)d + blockIdx.x*pitch_d);
		for(int k=threadIdx.x; k<n; k+=blockDim.x)
		{
			if(k == i) //If k is the source vertex
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
		__shared__ int done; //Did we finish on the first iteration?

		if(j == 0)
		{
			Q_row[0] = i;
			Q_len = 1;
			Q2_len = 0;
			done = false;
		}
		__syncthreads();

		//Do first iteration seperately, since we already know the edges to traverse
		for(int r=threadIdx.x+R[i]; r<R[i+1]; r+=blockDim.x)
		{
			int w = C[r];
			//Assuming no duplicate/self edges in the graph - each value of w is unique, so no need for atomics
			if(d_row[w] == INT_MAX)
			{
				d_row[w] = 1;
				int t = atomicAdd(&Q2_len,1);
				Q2_row[t] = w;
			}
		}
		__syncthreads();

		if(Q2_len == 0)
		{
			done = true;
		}
		else
		{
			for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
			{
				Q_row[kk] = Q2_row[kk];
			}
			__syncthreads();

			if(j == 0)
			{
				Q_len = Q2_len;
				Q2_len = 0;
			}
		}
		__syncthreads();

		while(!done)
		{
			for(int k=threadIdx.x; k<Q_len; k+=blockDim.x)
			{
				int v = Q_row[k];
				for(int r=R[v]; r<R[v+1]; r++)
				{
					int w = C[r];
					//Use atomicCAS() to prevent duplicate queue entries
					if(atomicCAS(&d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
					{
						int t = atomicAdd(&Q2_len,1);
						Q2_row[t] = w;
					}
				}
			}
			__syncthreads();

			if(Q2_len == 0)
			{
				break;
			}
			else
			{
				for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
				{
					Q_row[kk] = Q2_row[kk];
				}
				__syncthreads();

				if(j == 0)
				{
					Q_len = Q2_len;
					Q2_len = 0;
				}
				__syncthreads();
			}
		}
	}
}
