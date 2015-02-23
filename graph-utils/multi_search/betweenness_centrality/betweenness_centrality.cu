#include "betweenness_centrality.cuh"

//TODO: Return reference
void betweenness_centrality_setup(const device_graph &g, int start, int end, std::vector< std::vector<float> > &delta_h)
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

	//Memory specific to Load-Balancing Search
	int *edge_counts_d, *scanned_edges_d, *LBS_d;
	thrust::device_vector<int> edge_frontier_size_d(dimGrid.x,0);

	checkCudaErrors(cudaMallocPitch((void**)&edge_counts_d,&p.edge_counts,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&scanned_edges_d,&p.scanned_edges,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&LBS_d,&p.LBS,sizeof(int)*g.m,dimGrid.x));

        size_t GPU_memory_requirement = sizeof(int)*g.n*sources_to_store + 4*sizeof(int)*g.n*dimGrid.x + sizeof(int)*(g.n+1) + sizeof(int)*(g.m) + sizeof(unsigned long long)*g.n*sources_to_store + sizeof(float)*g.n*sources_to_store + sizeof(float)*g.n + sizeof(int)*sources_to_store + 2*sizeof(int)*g.n*sources_to_store + sizeof(int)*g.m*sources_to_store;
        std::cout << "BC memory requirement: " << GPU_memory_requirement/(1 << 20) << " MB" << std::endl;

	betweenness_centrality<<<dimGrid,896>>>(thrust::raw_pointer_cast(g.R.data()),thrust::raw_pointer_cast(g.C.data()),thrust::raw_pointer_cast(g.F.data()),g.n,g.m,d_d,sigma_d,delta_d,thrust::raw_pointer_cast(bc_d.data()),Q_d,Q2_d,S_d,endpoints_d,thrust::raw_pointer_cast(edge_frontier_size_d.data()),edge_counts_d,scanned_edges_d,LBS_d,p,start,end);
	checkCudaErrors(cudaPeekAtLastError());

        //std::vector< std::vector<float> > delta_h;
        transfer_result(g,delta_d,p.delta,sources_to_store,delta_h);

	//Free algorithm-specific memory
	checkCudaErrors(cudaFree(LBS_d));
	checkCudaErrors(cudaFree(scanned_edges_d));
	checkCudaErrors(cudaFree(edge_counts_d));
	checkCudaErrors(cudaFree(endpoints_d));
	checkCudaErrors(cudaFree(S_d));
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(delta_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(d_d));
	float time = end_clock(start_event,end_event);

	std::cout << "Time for shuffle-based BC: " << std::setprecision(9) << time << " s" << std::endl;
}

