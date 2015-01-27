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
	//size_t pitch_d, pitch_Q, pitch_Q2;
	pitch p;
	cudaEvent_t start_event, end_event;

	//Allocate algorithm-specific memory
	start_clock(start_event,end_event);
	checkCudaErrors(cudaMallocPitch((void**)&d_d,&p.d,sizeof(int)*g.n,sources_to_store));
	checkCudaErrors(cudaMallocPitch((void**)&Q_d,&p.Q,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&p.Q2,sizeof(int)*g.n,dimGrid.x));

        size_t GPU_memory_requirement = sizeof(int)*g.n*sources_to_store + 2*sizeof(int)*g.n*dimGrid.x + sizeof(int)*(g.n+1) + sizeof(int)*(g.m);
        std::cout << "Shuffle based memory requirement: " << GPU_memory_requirement/(1 << 20) << " MB" << std::endl;

	multi_search_shuffle_based<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),g.n,d_d,Q_d,Q2_d,p,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        std::vector< std::vector<int> > d_host_vector;
        transfer_result(g,d_d,p.d,sources_to_store,d_host_vector);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for shuffle-based neighbor gathering: " << std::setprecision(9) << time << " s" << std::endl;

	return d_host_vector;
}

//Wrappers
__global__ void multi_search_shuffle_based(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, const pitch p, const int start, const int end)
{
        auto null_lamb_1 = [](int){};
	auto null_lamb_2 = [](int,int){};
	auto null_lamb_3 = [](int*,int,int){};
        multi_search(R,C,n,d,Q,Q2,p,start,end,null_lamb_1,null_lamb_2,null_lamb_3);
}

__global__ void diameter_sampling(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, int *max, const pitch p, const int start, const int end)
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

        multi_search(R,C,n,d,Q,Q2,p,start,end,max_lamb,null_lamb_1,null_lamb_2);
}

__global__ void all_pairs_shortest_paths(const int *R, const int *C, const int n, int *d, unsigned long long *sigma, int *Q, int *Q2, const pitch p, const int start, const int end)
{
	auto null_lamb = [](int){};

	auto init_sigma_row = [sigma,p] (int k, int i)
	{
		auto sigma_row = get_row(sigma,p.sigma); //In theory this needs to be called every iteration on i if we're going to store all of the results
		if(k == i)
		{
			sigma_row[k] = 1;
		}
		else
		{
			sigma_row[k] = 0;
		}
	};

	auto update_sigma_row = [sigma,p] (int *d_row, int v, int w)
	{
		if(d_row[w] == (d_row[v]+1))
		{
			auto sigma_row = get_row(sigma,p.sigma); //In theory this needs to be called every iteration on i if we're going to store all of the results
			atomicAdd(&sigma_row[w],sigma_row[v]);
		}
	};

	multi_search(R,C,n,d,Q,Q2,p,start,end,null_lamb,init_sigma_row,update_sigma_row);
}

//TODO: Make sure bc is memset to 0 before calling this function
__global__ void betweenness_centrality(const int *R, const int *C, const int n, int *d, unsigned long long *sigma, float *delta, float *bc, int *Q, int *Q2, int *S, int *endpoints, const pitch p, const int start, const int end)
{
	auto init_sigma_delta = [p,sigma,delta] (int k, int i)
	{
		unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*p.sigma);
		float *delta_row = (float *)((char*)delta + blockIdx.x*p.delta);
		if(k == i)
		{
			sigma_row[k] = 1;
		}
		else
		{
			sigma_row[k] = 0;
		}
		delta_row[k] = 0;
	};

	__shared__ int S_len;
	__shared__ int endpoints_len;
	__shared__ int current_depth;
	auto init_S_endpoints = [p,S,endpoints,S_len,endpoints_len] ()
	{
		int *S_row = (int*)((char*)S + blockIdx.x*p.S);
	};

}
