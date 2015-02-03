#include "betweenness_centrality.cuh"

//TODO: Return reference
std::vector< std::vector<float> > betweenness_centrality_setup(const device_graph &g, int start, int end)
{
	//For now, use "standard" grid/block sizes. These can be tuned later on.
	dim3 dimGrid, dimBlock;
        //Returns number of source vertices to store for verification purposes
        size_t sources_to_store = configure_grid(dimGrid,dimBlock,start,end);

	//Device pointers
	int *d_d, *Q_d, *Q2_d, *S_d, *endpoints_d;
	unsigned long long *sigma_d;
	float *delta_d;
	pitch p;
	cudaEvent_t start_event, end_event;

	//Allocate algorithm-specific memory
	start_clock(start_event,end_event);
	checkCudaErrors(cudaMallocPitch((void**)&d_d,&p.d,sizeof(int)*g.n,sources_to_store));
	checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&p.sigma,sizeof(unsigned long long)*g.n,sources_to_store));
	checkCudaErrors(cudaMallocPitch((void**)&delta_d,&p.delta,sizeof(float)*g.n,sources_to_store));
	checkCudaErrors(cudaMallocPitch((void**)&Q_d,&p.Q,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&p.Q2,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&S_d,&p.S,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&endpoints_d,&p.endpoints,sizeof(int)*g.n,dimGrid.x));
	thrust::device_vector<float> bc_d(g.n,0);

        size_t GPU_memory_requirement = sizeof(int)*g.n*sources_to_store + 4*sizeof(int)*g.n*dimGrid.x + sizeof(int)*(g.n+1) + sizeof(int)*(g.m) + sizeof(unsigned long long)*g.n*sources_to_store + sizeof(float)*g.n*sources_to_store + sizeof(float)*g.n;
        std::cout << "BC memory requirement: " << GPU_memory_requirement/(1 << 20) << " MB" << std::endl;

	betweenness_centrality<<<dimGrid,1024>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),g.n,d_d,sigma_d,delta_d,thrust::raw_pointer_cast(bc_d.data()),Q_d,Q2_d,S_d,endpoints_d,p,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        std::vector< std::vector<float> > delta_h;
        transfer_result(g,delta_d,p.delta,sources_to_store,delta_h);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(endpoints_d));
	checkCudaErrors(cudaFree(S_d));
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(delta_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for shuffle-based BC: " << std::setprecision(9) << time << " s" << std::endl;

	return delta_h;
}

