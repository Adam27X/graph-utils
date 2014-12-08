#include "util_device.cuh"

//Note: Times are returned in seconds
void start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&end));
	checkCudaErrors(cudaEventRecord(start,0));
}

float end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	checkCudaErrors(cudaEventRecord(end,0));
	checkCudaErrors(cudaEventSynchronize(end));
	checkCudaErrors(cudaEventElapsedTime(&time,start,end));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(end));

	return time/(float)1000;
}

void choose_device(const program_options &op)
{
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
	cudaDeviceProp prop;

	if(op.device == -1)
	{
		//Choose the device with the greatest memory BW
		double maxMemBandwidth = 0;
		int bestdev = 0;
		for(int i=0; i<count; i++)
		{
			checkCudaErrors(cudaGetDeviceProperties(&prop,i));
			double memBandwidth = (prop.memoryClockRate * 1000.0) * (prop.memoryBusWidth / 8 * 2) / 1.0e9;
			if(memBandwidth > maxMemBandwidth)
			{
				maxMemBandwidth = memBandwidth;
				bestdev = i;
			}
		}
		checkCudaErrors(cudaSetDevice(bestdev));
		checkCudaErrors(cudaGetDeviceProperties(&prop,bestdev));
	}
	else if((op.device < -1) || (op.device >= count))
	{
		std::cerr << "Invalid device argument. Valid devices on this machine range from 0 to " << count-1 << std::endl;
		exit(-1);
	}
	else
	{
		checkCudaErrors(cudaSetDevice(op.device));
		checkCudaErrors(cudaGetDeviceProperties(&prop,op.device));
	}

	size_t free_mem, total_mem;
	checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
	double memBandwidth = (prop.memoryClockRate * 1000.0) * (prop.memoryBusWidth / 8 * 2) / 1.0e9;
	std::cout << "Chosen Device: " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Number of Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
	std::cout << "Size of Global Memory: " << (total_mem/(double)(1 << 30)) << " GB" << std::endl;
	std::cout << "Memory Bandwidth: " << memBandwidth << " GB/s " << std::endl;
}
