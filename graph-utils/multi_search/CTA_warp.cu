#include "CTA_warp.cuh"
#include "common.cuh"

std::vector< std::vector<int> > multi_search_CTA_warp_based_setup(const device_graph &g, int start, int end)
{
	//For now, use "standard" grid/block sizes. These can be tuned later on.
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

        size_t GPU_memory_requirement = sizeof(int)*g.n*sources_to_store + 2*sizeof(int)*g.n*dimGrid.x + sizeof(int)*(g.n+1) + sizeof(int)*(g.m);
        std::cout << "CTA+warp memory requirement: " << GPU_memory_requirement/(1 << 20) << " MB" << std::endl;

	multi_search_CTA_warp_based<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),g.n,d_d,pitch_d,Q_d,pitch_Q,Q2_d,pitch_Q2,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        std::vector< std::vector<int> > d_host_vector;
        transfer_result(g,d_d,pitch_d,sources_to_store,d_host_vector);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for CTA+warp neighbor gathering: " << std::setprecision(9) << time << " s" << std::endl;

	return d_host_vector;
}

__global__ void multi_search_CTA_warp_based(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end)
{
	const int degree_threshold = 2048;
	int j = threadIdx.x;
	int warp_id = threadIdx.x/32;
	int lane_id = threadIdx.x & 0x1f;
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
			volatile __shared__ int comm[32][4]; //32 is the number of warps
			__shared__ int CTA_comm[4];
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

			//If a thread's adjlist is really large, have the entire CTA process it. 
			while(1)
			{
				__shared__ bool high_degree;
				if(threadIdx.x == 0)
				{
					high_degree = false;
				}
				__syncthreads();
				for(int m=threadIdx.x; m<Q_len; m+=blockDim.x)
				{
					if(r_end-r > degree_threshold)
					{
						high_degree = true;
						CTA_comm[0] = threadIdx.x;
					}
				}
				__syncthreads();
				while(high_degree) 
				{
					if(CTA_comm[0] == threadIdx.x)
					{
						CTA_comm[1] = r;
						CTA_comm[2] = r_end;
						CTA_comm[3] = v;
						r = 0; //Same thread cannot win twice
						r_end = 0; 
					}
					__syncthreads();
					
					int r_gather = CTA_comm[1] + threadIdx.x;
					int r_gather_end = CTA_comm[2];
					int v_new = CTA_comm[3];
					while(r_gather < r_gather_end)
					{
						volatile int w = C[r_gather];
						//Assuming no duplicate/self-edges in the graph, no atomics needed
						if(d_row[w] == INT_MAX)
						{
							d_row[w] = d_row[v_new]+1;
							int t = atomicAdd(&Q2_len,1);
							Q2_row[t] = w;
						}
						r_gather += blockDim.x;
					}
					__syncthreads();
					
					//See if another loop iteration is needed. This would be cleaner in a device function
					if(threadIdx.x == 0)
					{
						high_degree = false;
					}
					__syncthreads();
					for(int m=threadIdx.x; m<Q_len; m+=blockDim.x)
					{
						if(r_end-r > degree_threshold)
						{
							high_degree = true;
							CTA_comm[0] = threadIdx.x;
						}
					}
					__syncthreads();
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

				if((k-threadIdx.x) >= Q_len) //If thread 0 doesn't have work, the entire block is done
				{
					break;
				}
			}

			//Next, use the warp-based approach
			k = threadIdx.x; //Go back to the beginning of the queue

			if(k < Q_len)
			{	
				v = Q_row[k];
				r = R[v];
				r_end = R[v+1];
				if(r_end-r > degree_threshold) //These were already taken care of by the entire CTA
				{
					r = 0;
					r_end = 0;
				}
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
					if(r_end-r)
					{
						comm[warp_id][0] = lane_id;
					}

					//Winner describes adjlist
					if(comm[warp_id][0] == lane_id)
					{
						comm[warp_id][1] = r;
						comm[warp_id][2] = r_end;
						comm[warp_id][3] = v;
						r = 0; //Same thread cannot win twice
						r_end = 0;
					}

					//Strip mine winner's adjlist
					int r_gather = comm[warp_id][1] + lane_id;
					int r_gather_end = comm[warp_id][2];
					int v_new = comm[warp_id][3];
					while(r_gather < r_gather_end)
					{
						volatile int w = C[r_gather];
						//Assuming no duplicate/self-edges in the graph, no atomics needed
						if(d_row[w] == INT_MAX)
						{
							d_row[w] = d_row[v_new]+1;
							int t = atomicAdd(&Q2_len,1);
							Q2_row[t] = w;
						}
						r_gather += WARP_SIZE;
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

				if((k-threadIdx.x) >= Q_len) //If thread 0 doesn't have work, the entire warp is done
				{
					break;
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
