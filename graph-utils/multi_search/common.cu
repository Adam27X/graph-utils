#include "common.cuh"

size_t configure_grid(dim3 &dimGrid, dim3 &dimBlock, int start, int end)
{
	cudaDeviceProp prop;
	int dev;
	checkCudaErrors(cudaGetDevice(&dev));
	checkCudaErrors(cudaGetDeviceProperties(&prop,dev));
	dimGrid.x = prop.multiProcessorCount;
	dimGrid.y = 1;
	dimGrid.z = 1;

	dimBlock.x = prop.maxThreadsPerBlock;
	dimBlock.y = 1;
	dimBlock.z = 1;

        size_t sources_to_store;
        if((end-start) < dimGrid.x)
        {
                sources_to_store = end-start;
        }
        else
        {
                sources_to_store = dimGrid.x;
        }

	return sources_to_store;	
}

void transfer_result(const device_graph &g, int *d_d, size_t pitch_d, size_t sources_to_store, std::vector< std::vector<int> > &d_host_vector)
{
        int *d_host_array = new int[g.n*sources_to_store];
        checkCudaErrors(cudaMemcpy2D(d_host_array,sizeof(int)*g.n,d_d,pitch_d,sizeof(int)*g.n,sources_to_store,cudaMemcpyDeviceToHost));
        d_host_vector.resize(sources_to_store);
        for(int i=0; i<sources_to_store; i++)
        {
                d_host_vector[i].resize(g.n);
                for(int j=0; j<g.n; j++)
                {
                        d_host_vector[i][j] = d_host_array[i*g.n + j];
                }
        }
        delete[] d_host_array;
}

