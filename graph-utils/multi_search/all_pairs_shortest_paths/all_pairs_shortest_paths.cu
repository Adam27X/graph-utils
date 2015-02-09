#include "all_pairs_shortest_paths.cuh"

//TODO: Pass in a reference vector and return void
std::vector< std::vector<unsigned long long> > all_pairs_shortest_paths_setup(const device_graph &g, int start, int end)
{
	//For now, use "standard" grid/block sizes. These can be tuned later on.
	dim3 dimGrid, dimBlock;
        //Returns number of source vertices to store for verification purposes
        size_t sources_to_store = configure_grid(dimGrid,dimBlock,start,end);

	//Device pointers
	int *d_d, *Q_d, *Q2_d;
	unsigned long long *sigma_d;
	//size_t pitch_d, pitch_Q, pitch_Q2, pitch_sigma;
	pitch p;
	cudaEvent_t start_event, end_event;

	//Allocate algorithm-specific memory
	start_clock(start_event,end_event);
	checkCudaErrors(cudaMallocPitch((void**)&d_d,&p.d,sizeof(int)*g.n,sources_to_store));
	checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&p.sigma,sizeof(unsigned long long)*g.n,sources_to_store));
	checkCudaErrors(cudaMallocPitch((void**)&Q_d,&p.Q,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&p.Q2,sizeof(int)*g.n,dimGrid.x));

        size_t GPU_memory_requirement = sizeof(int)*g.n*sources_to_store + 2*sizeof(int)*g.n*dimGrid.x + sizeof(int)*(g.n+1) + sizeof(int)*(g.m) + sizeof(unsigned long long)*g.n*sources_to_store;
        std::cout << "APSP memory requirement: " << GPU_memory_requirement/(1 << 20) << " MB" << std::endl;

	all_pairs_shortest_paths<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),g.n,d_d,sigma_d,Q_d,Q2_d,p,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        std::vector< std::vector<unsigned long long> > sigma_host_vector;
        transfer_result(g,sigma_d,p.sigma,sources_to_store,sigma_host_vector);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for shuffle-based APSP: " << std::setprecision(9) << time << " s" << std::endl;

	return sigma_host_vector;
}

