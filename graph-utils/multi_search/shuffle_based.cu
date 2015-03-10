#include "shuffle_based.cuh"
#include "common.cuh"

//TODO: Pass in a reference vector and return void
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


//TODO: Move lambdas to a global space so that they can be reused (and rename them appropriately)
//Wrappers
__global__ void multi_search_shuffle_based(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, const pitch p, const int start, const int end)
{
        auto null_lamb_1 = [](int){}; //A bit ugly, but it works
	auto null_lamb_2 = [](int,int){};
	auto null_lamb_3 = [](int*,int){};
	auto null_lamb_4 = [](int*,int,int){};
	auto null_lamb_5 = [](int*,int,int,int){};
        multi_search(R,C,n,d,Q,Q2,p,start,end,null_lamb_1,null_lamb_2,null_lamb_4,null_lamb_1,null_lamb_3,null_lamb_1,null_lamb_5);
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

	auto null_lamb_1 = [](int){};
	auto null_lamb_2 = [](int,int){};
	auto null_lamb_3 = [](int*,int){};
	auto null_lamb_4 = [](int*,int,int){};
	auto null_lamb_5 = [](int*,int,int,int){};

        multi_search(R,C,n,d,Q,Q2,p,start,end,max_lamb,null_lamb_2,null_lamb_4,null_lamb_1,null_lamb_3,null_lamb_1,null_lamb_5);
}

__global__ void count_shortest_paths(const int *R, const int *C, const int n, int *d, unsigned long long *sigma, int *Q, int *Q2, const pitch p, const int start, const int end)
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

	auto null_lamb_1 = [](int){};
	auto null_lamb_2 = [](int,int){};
	auto null_lamb_3 = [](int*,int){};
	auto null_lamb_4 = [](int*,int,int){};
	auto null_lamb_5 = [](int*,int,int,int){};

	multi_search(R,C,n,d,Q,Q2,p,start,end,null_lamb_1,init_sigma_row,update_sigma_row,null_lamb_1,null_lamb_3,null_lamb_1,null_lamb_5);
}

#define DSAMPLE 210
__global__ void betweenness_centrality(const int *R, const int *C, const int *F, const int n, const int m, int *d, unsigned long long *sigma, float *delta, float *bc, int *Q, int *Q2, int *S, int *endpoints, const pitch p, const int start, const int end, int *dep_accum)
{
	__shared__ unsigned long long *sigma_row;
	__shared__ float *delta_row;
	__shared__ int *S_row;
	__shared__ int *endpoints_row;
	__shared__ int S_len;
	__shared__ int endpoints_len;
	__shared__ int current_depth;

	auto init_sigma_delta = [&sigma_row,&delta_row,p,sigma,delta,bc] (int k, int i)
	{
		sigma_row = get_row(sigma,p.sigma); //In theory this needs to be called every iteration on i if we're going to store all of the results
		delta_row = get_row(delta,p.delta);
		if(k == i)
		{
			sigma_row[k] = 1;
		}
		else
		{
			sigma_row[k] = 0;
		}
		delta_row[k] = 0;
		bc[k] = 0;
	};

	auto init_S_endpoints = [p,S,endpoints,&S_len,&endpoints_len,&S_row,&endpoints_row] (int i)
	{
		S_row = get_row(S,p.S);
		endpoints_row = get_row(endpoints,p.endpoints);
		S_row[0] = i;
		S_len = 1;
		endpoints_row[0] = 0;
		endpoints_row[1] = 1;
		endpoints_len = 2;
	};

	auto update_sigma_row = [&sigma_row] (int *d_row, int v, int w)
	{
		if(d_row[w] == (d_row[v]+1))
		{
			atomicAdd(&sigma_row[w],sigma_row[v]);
		}
	};

	auto insert_stack = [&S_len,&S_row] (int *Q2_row, int kk)
	{
		S_row[kk+S_len] = Q2_row[kk];
	};

	auto update_endpoints = [&endpoints_len,&S_len,&endpoints_row] (int Q2_len)
	{
		endpoints_row[endpoints_len] = endpoints_row[endpoints_len-1] + Q2_len;
		endpoints_len++;
		S_len += Q2_len;
	};

	//TODO: Get rid of some registers here too, if possible. Or, tune differently.
	auto dependency_accumulation = [&S_len,&endpoints_len,&current_depth,&S_row,&endpoints_row,&sigma_row,&delta_row,bc,R,C,n] (int *d_row, int i, int j, int lane_id)
	{
		//Set current depth
		if(j == 0)
		{
			current_depth = d_row[S_row[S_len-1]] - 1;
		}	
		__syncthreads();
		
		while(current_depth > 0) 
		{
			int w, r, r_end;
			int k = threadIdx.x;
			__shared__ int depth_size;
		       
			if(threadIdx.x == 0)
			{
				depth_size = endpoints_row[current_depth+1]-endpoints_row[current_depth];
			}
			__syncthreads();

			if(k < depth_size)
			{
				w = S_row[k+endpoints_row[current_depth]];
				r = R[w];
				r_end = R[w+1];
			}
			else
			{
				w = -1;
				r = 0;
				r_end = 0;
			}

			while(1)
			{
				while(__any(r_end-r))
				{
					//Vie for control of warp
					int winner = race_and_resolve_warp(r_end-r);

					//Strip mine winner's adjlist
					int r_gather = __shfl(r,winner) + lane_id;
					int r_gather_end = __shfl(r_end,winner);
					int w_new = __shfl(w,winner);
					unsigned long long sw = sigma_row[w_new];
					float dsw = 0.0;
					while(r_gather < r_gather_end)
					{
						int v = C[r_gather];
						if(d_row[v] == (d_row[w_new]+1))
						{
							dsw += (sw/(float)sigma_row[v])*(1.0f+delta_row[v]);
						}
						r_gather += WARP_SIZE;
					}

					typedef cub::WarpReduce<float> WarpReduceFloat;
					__shared__ typename WarpReduceFloat::TempStorage temp_storage[32]; //Temporary storage for each warp - 32 is max warps per block
					float dsw_agg = WarpReduceFloat(temp_storage[threadIdx.x/32]).Sum(dsw);
					if(lane_id == 0)
					{
						delta_row[w_new] += dsw_agg;
					}

					if(winner == lane_id) //Same thread cannot win twice
					{
						r = 0;
						r_end = 0;
					}
				}

				k+=blockDim.x;
				if(k < depth_size)
				{
					w = S_row[k+endpoints_row[current_depth]];
					r = R[w];
					r_end = R[w+1];
				}
				else
				{
					w = -1;
					r = 0;
					r_end = 0;
				}

				if((k-threadIdx.x) >= depth_size) //If thread 0 has no work, the entire block is done
				{
					break;
				}
			}
			__syncthreads();
			if(j == 0)
			{
				current_depth--;
			}
			__syncthreads();
		}

		for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
		{
			atomicAdd(&bc[kk],delta_row[kk]); //delta_row[i] is guaranteed to be zero. 
		}
	};

	//See if the old method is preferential for the dependency accumulation
	//To make verification work properly, DIAMETER_SAMPLES must be a multiple of the number of SMs
	const int diameter_samples = DSAMPLE; //FIXME: Have a portable way to introduce such a number. LCM(14,15) = 210, so this value works both for the Titan and the K40
	__shared__ int diameters[diameter_samples];
	auto dependency_accumulation_work_eff = [&S_len,&endpoints_len,&current_depth,&S_row,&endpoints_row,&sigma_row,&delta_row,bc,R,C,n,diameter_samples,&diameters] (int *d_row, int i, int j, int lane_id)
	{
		//Set current depth
		if(j == 0)
		{
			current_depth = d_row[S_row[S_len-1]] - 1;
			if(i < diameter_samples)
			{
				diameters[i] = current_depth;
			}
		}
		__syncthreads();

		while(current_depth > 0)
		{
			for(int kk=threadIdx.x+endpoints_row[current_depth]; kk<endpoints_row[current_depth+1]; kk+=blockDim.x)
			{
				int w = S_row[kk];
				float dsw = 0;
				float sw = (float)sigma_row[w];
				for(int z=R[w]; z<R[w+1]; z++)
				{
					int v = C[z];
					if(d_row[v] == (d_row[w]+1))
					{	
						dsw += (sw/(float)sigma_row[v])*(1.0f+delta_row[v]);
					}
				}
				delta_row[w] = dsw;	
			}
			__syncthreads();
			if(j == 0)
			{
				current_depth--;
			}
			__syncthreads();
		}

		for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
		{
			atomicAdd(&bc[kk],delta_row[kk]);
		}
	};

	auto dependency_accumulation_edge_par = [&S_len,&endpoints_len,&current_depth,&S_row,&endpoints_row,&sigma_row,&delta_row,bc,R,C,F,n,m] (int *d_row, int i, int j, int lane_id)
	{
		//Set current depth
		if(j == 0)
		{
			current_depth = d_row[S_row[S_len-1]] - 1;
		}
		__syncthreads();

		while(current_depth > 0)
		{
			for(int kk=threadIdx.x; kk<m; kk+=blockDim.x)
			{
				int w = F[kk];
				if(d_row[w] == current_depth)
				{
					int v = C[kk];
					if(d_row[v] == (current_depth+1))
					{
						float sw = (float)sigma_row[w];
						float change = (sw/(float)sigma_row[v])*(1.0f+delta_row[v]);
						atomicAdd(&delta_row[w],change);
					}
				}
			}
			__syncthreads();
			if(j == 0)
			{
				current_depth--;
			}
			__syncthreads();
		}

		for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
		{
			atomicAdd(&bc[kk],delta_row[kk]);
		}
	};

	/*auto dependency_accumulation_lbs = [p,&S_len,&endpoints_len,&current_depth,d,bc,R,C,n,edge_frontier_size,edge_counts,scanned_edges,LBS,&S_row,&endpoints_row,&sigma_row,&delta_row] (int *d_row, int i, int j, int lane_id)
	{
		//Set current depth
		__shared__ int *edge_counts_row;
		__shared__ int *scanned_edges_row;
		__shared__ int *LBS_row;
		if(j == 0)
		{
			edge_counts_row = get_row(edge_counts,p.edge_counts);
			scanned_edges_row = get_row(scanned_edges,p.scanned_edges);
			LBS_row = get_row(LBS,p.LBS);
			current_depth = d_row[S_row[S_len-1]] - 1;
		}
		__syncthreads();

		while(current_depth > 0)
		{
			__shared__ int vertex_frontier_size;
		       	if(j == 0)
			{
				vertex_frontier_size = endpoints_row[current_depth+1]-endpoints_row[current_depth];
			}
			__syncthreads();
			//Fill in edge counts
			for(int kk=threadIdx.x; kk<vertex_frontier_size; kk+=blockDim.x)
			{
				int w = S_row[kk+endpoints_row[current_depth]]; 
				edge_counts_row[kk] = R[w+1]-R[w]; 
			}
			__syncthreads();
			load_balance_search_block(vertex_frontier_size,&edge_frontier_size[blockIdx.x],edge_counts_row,scanned_edges_row,LBS_row);
			__syncthreads();
			for(int kk=threadIdx.x; kk<edge_frontier_size[blockIdx.x]; kk+=blockDim.x)
			{
				int w = S_row[LBS_row[kk]+endpoints_row[current_depth]];
				int rank = kk - scanned_edges_row[LBS_row[kk]];
				int v = C[R[w]+rank];
				if(d_row[v] == (d_row[w]+1))
				{
					float change = ((float)sigma_row[w]/(float)sigma_row[v])*(1.0f+delta_row[v]);
					atomicAdd(&delta_row[w],change);
				}
			}
			__syncthreads();
			if(j == 0)
			{
				current_depth--;
			}
			__syncthreads();
		}

		for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
		{
			atomicAdd(&bc[kk],delta_row[kk]);
		}
	};*/
        
	auto null_lamb_1 = [](int){}; //getMax

	if(n > DSAMPLE)
	{
		//Two calls: One that goes from [start,DIAMETER_SAMPLES) and another from [DIAMETER_SAMPLES,end). 
		multi_search(R,C,n,d,Q,Q2,p,start,diameter_samples,null_lamb_1,init_sigma_delta,update_sigma_row,init_S_endpoints,insert_stack,update_endpoints,dependency_accumulation_work_eff);
		__shared__ int diameter;
		if(threadIdx.x==0)
		{
			diameter = 0;
		}
		__syncthreads();
		for(int z=threadIdx.x; z<diameter_samples/2; z+=blockDim.x) //Each block computes this independently. Choose diameter_samples/2 to ensure blocks have valid data.
		{
			atomicMax(&diameter,diameters[z]);	
		}
		__syncthreads();
		__shared__ int log2n;
		if(threadIdx.x == 0)
		{
			log2n = 0;
			int tempn = n;
			while(tempn >>= 1)
			{
				++log2n;
			}
			dep_accum[0] = diameter;
		}
		__syncthreads();
		if(diameter < 4*log2n)
		{
			multi_search(R,C,n,d,Q,Q2,p,diameter_samples,end,null_lamb_1,init_sigma_delta,update_sigma_row,init_S_endpoints,insert_stack,update_endpoints,dependency_accumulation_edge_par);
		}
		else
		{
			multi_search(R,C,n,d,Q,Q2,p,diameter_samples,end,null_lamb_1,init_sigma_delta,update_sigma_row,init_S_endpoints,insert_stack,update_endpoints,dependency_accumulation_work_eff);
		}
	}
	else
	{
		multi_search(R,C,n,d,Q,Q2,p,start,end,null_lamb_1,init_sigma_delta,update_sigma_row,init_S_endpoints,insert_stack,update_endpoints,dependency_accumulation_work_eff);
	}
}

__global__ void transitive_closure(const int *R, const int *C, const int n, int *d, int *Q, int *Q2, const pitch p, const int start, const int end)
{
        auto null_lamb_1 = [](int){}; //A bit ugly, but it works
        auto null_lamb_2 = [](int,int){};
        auto null_lamb_3 = [](int*,int){};
        auto null_lamb_4 = [](int*,int,int){};
        auto null_lamb_5 = [](int*,int,int,int){};
        multi_search(R,C,n,d,Q,Q2,p,start,end,null_lamb_1,null_lamb_2,null_lamb_4,null_lamb_1,null_lamb_3,null_lamb_1,null_lamb_5);
}
