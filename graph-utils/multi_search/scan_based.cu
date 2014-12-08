#include "scan_based.cuh"
#include "common.cuh"
#include <cub/block/block_scan.cuh>

std::vector< std::vector<int> > multi_search_scan_based_setup(device_graph &g, int start, int end)
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

	multi_search_scan_based<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),g.n,d_d,pitch_d,Q_d,pitch_Q,Q2_d,pitch_Q2,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        std::vector< std::vector<int> > d_host_vector;
        transfer_result(g,d_d,pitch_d,sources_to_store,d_host_vector);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for scan-based neighbor gathering: " << time << " s" << std::endl;

	return d_host_vector;
}

__global__ void multi_search_scan_based(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end)
{
	int j = threadIdx.x;
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
		__shared__ int current_depth;

		if(j == 0)
		{
			Q_row[0] = i;
			Q_len = 1;
			Q2_len = 0;
			current_depth = 0;
		}
		__syncthreads();

		while(1)
		{
			//Listing 10: Scan-based neighbor gathering
			volatile __shared__ int comm[1024][2]; //1024 is the number of threads per CTA
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

			//Reserve gather offsets using cub::BlockScan
			//Specialize cub::BlockScan for a 1D block of 1024 threads on type int
			typedef cub::BlockScan<int,1024,cub::BLOCK_SCAN_RAKING,1,1,350> BlockScan;
			//Allocate shared memory for BlockScan
			__shared__ typename BlockScan::TempStorage temp_storage;
			//For now, scan the items for the blockDim.x (1024) queue items that are currently being inspected
			//TODO: It should be more efficient to have threads gather all of their queue elements at once
			int rsv_rank;
			int total;

			//Process fine-grained batches of adjlists
			while(1)
			{
				int cta_progress = 0;
				BlockScan(temp_storage).ExclusiveSum(r_end-r,rsv_rank,total);
				int remain;
				while((remain = total-cta_progress) > 0)
				{
					//Share batch of gather offsets
					while((rsv_rank < cta_progress + blockDim.x) && (r < r_end))
					{
						comm[rsv_rank-cta_progress][0] = r;
						comm[rsv_rank-cta_progress][1] = v;
						rsv_rank++;
						r++;
					}
					__syncthreads();

					//Gather batch of adjlist(s)
					int min_threads_remain = (remain < blockDim.x) ? remain : blockDim.x;
					if(threadIdx.x < min_threads_remain)
					{
						volatile int w = C[comm[threadIdx.x][0]];
						int v_new = comm[threadIdx.x][1];
						//Not sure if the v originally gathered by a thread will correspond to the parent of the neighbor that is found
						// so keep track of the current depth and use that number instead
						//if(atomicCAS(&d_row[w],INT_MAX,current_depth+1) == INT_MAX)
						if(atomicCAS(&d_row[w],INT_MAX,d_row[v_new]+1) == INT_MAX)
						{
							int t = atomicAdd(&Q2_len,1);
							Q2_row[t] = w;
						}
					}
					cta_progress += blockDim.x;
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
					current_depth++;
				}
				__syncthreads();
			}
		}
	}
}
