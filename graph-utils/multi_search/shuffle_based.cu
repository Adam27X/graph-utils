#include "shuffle_based.cuh"
#include "common.cuh"

std::vector< std::vector<int> > multi_search_shuffle_based_setup(const device_graph &g, int start, int end)
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
        std::cout << "Shuffle based memory requirement: " << GPU_memory_requirement/(1 << 20) << " MB" << std::endl;

	multi_search_shuffle_based<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),g.n,d_d,pitch_d,Q_d,pitch_Q,Q2_d,pitch_Q2,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        std::vector< std::vector<int> > d_host_vector;
        transfer_result(g,d_d,pitch_d,sources_to_store,d_host_vector);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for shuffle-based neighbor gathering: " << std::setprecision(9) << time << " s" << std::endl;

	return d_host_vector;
}

//Wrappers
__global__ void multi_search_shuffle_based(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end)
{
        auto null_lamb_1 = [](int){};
	auto null_lamb_2 = [](int,int){};
	auto null_lamb_3 = [](int*,int,int){};
        multi_search(R,C,n,d,pitch_d,Q,pitch_Q,Q2,pitch_Q2,start,end,null_lamb_1,null_lamb_2,null_lamb_3);
}

__global__ void diameter_sampling(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, int *max, const int start, const int end)
{
        auto max_lamb = [max](int v) //Using a separate variable for kinder syntax highlighting in vim
        {
                if(v != INT_MAX)
                {
                        atomicMax(max,v);
                }
        };

	auto null_lamb_1 = [](int,int){};
	auto null_lamb_2 = [](int*,int,int){};

        multi_search(R,C,n,d,pitch_d,Q,pitch_Q,Q2,pitch_Q2,start,end,max_lamb,null_lamb_1,null_lamb_2);
}

__global__ void all_pairs_shortest_paths(const int *R, const int *C, const int n, int *d, size_t pitch_d, unsigned long long *sigma, size_t pitch_sigma, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end)
{
	auto null_lamb = [](int){};

	auto init_sigma_row = [sigma,pitch_sigma] (int k, int i)
	{
		unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
		if(k == i)
		{
			sigma_row[k] = 1;
		}
		else
		{
			sigma_row[k] = 0;
		}
	};

	auto update_sigma_row = [sigma,pitch_sigma] (int *d_row, int v, int w)
	{
		if(d_row[w] == d_row[v]+1)
		{
			unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
			atomicAdd(&sigma_row[w],sigma_row[v]);
		}
	};

	multi_search(R,C,n,d,pitch_d,Q,pitch_Q,Q2,pitch_Q2,start,end,null_lamb,init_sigma_row,update_sigma_row);
}
