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

void choose_device(program_options &op)
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
	int runtime_version;
	checkCudaErrors(cudaRuntimeGetVersion(&runtime_version));
	std::string dev_name(prop.name);
	if(dev_name.find("Tesla") != std::string::npos)
	{
		op.isTesla = true;
	}

	std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;
	std::cout << "Chosen Device: " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Number of Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
	std::cout << "Size of Global Memory: " << (total_mem/(double)(1 << 30)) << " GB" << std::endl;
	std::cout << "Memory Bandwidth: " << memBandwidth << " GB/s " << std::endl;
}

/*void start_power_sample(program_options op, pthread_t &thread, long period)
{
        if(op.device != 0) //Could add an isTelsa flag, or use the NVML library directly to ensure that power can be measured from the GPU of interest
        {
                std::cerr << "Warning: Power can only be measured for Tesla GPUs." << std::endl;
        }
        else
        {
                //Spawn pthread for power measurement
                psample = new bool;
                *psample = true;
                pthread_create(&thread, NULL, power_sample, (void*)period);
                //std::cout << "Thread created." << std::endl;
                //std::cout << "Psample in main: " << *psample << std::endl;
        }
}*/

