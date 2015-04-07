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
//FIXME: This can be improved through the use a class with start/end member functions. 
// Such a class can contain psample and future members, making the user's life easier
double power_sample(int dev, long period);
extern bool *psample;
//Note: period is the sampling period in milliseconds
void start_power_sample(const program_options &op, std::future<double> &f, long period);
double end_power_sample(const program_options &op, std::future<double> &f);
