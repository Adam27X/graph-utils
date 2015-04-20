#pragma once

#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <nvml.h>
#include <future>
#include "util.h"

#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{   
    if (cudaSuccess != err)
    {   
        std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err) << " from file " << file  << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#ifndef checkNVMLErrors
#define checkNVMLErrors(err) __checkNVMLErrors (err, __FILE__, __LINE__)

inline void __checkNVMLErrors(nvmlReturn_t result, const char *file, const int line)
{
        if(result!=NVML_SUCCESS)
        {
                std::cerr << "NVML Error = " << result << ": " << nvmlErrorString(result) << " from file " << file << ", line " << line << std::endl;
                exit(EXIT_FAILURE);
        }
}
#endif

//Timing routines
void start_clock(cudaEvent_t &start, cudaEvent_t &end);
float end_clock(cudaEvent_t &start, cudaEvent_t &end);

void choose_device(program_options &op);

//Power sampling routines
class power_measurer
{
public:
	power_measurer(long p, const program_options &op) : period(p), device(op.device), isTesla(op.isTesla), psample(false) { }
	void start_power_sample();
	double end_power_sample();

private:
	int device; //GPU that we will attempt to sample power from
	bool isTesla; //Is the GPU we're sampling from a Tesla GPU?
	long period; //Sampling period
	bool psample;
	std::future<double> f; //Future used to asynchronously measure power
	double power_sample(int dev, long period); //Function called be async thread
	power_measurer(); //Disable default construction and force the user to supply a period and program options struct
};
