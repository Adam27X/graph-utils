#pragma once

#include "device_graph.h"
#include "util_device.cuh"
#include <iomanip>

//For now, use <<<# of SMs,max threads per block>>>
//Can autotune this later
size_t configure_grid(dim3 &dimGrid, dim3 &dimBlock, int start, int end);

//Transfer result to host. Use CUDA library calls to copy into a C-style array and then move that to a vector for convenience.
//Generic function to take a 2D result on the device and store it into a 2D std::vector
void transfer_result(const device_graph &g, int *d_d, size_t pitch_d, size_t sources_to_store, std::vector< std::vector<int> > &d_host_vector);
void transfer_result(const device_graph &g, unsigned long long *d_d, size_t pitch_d, size_t sources_to_store, std::vector< std::vector<unsigned long long> > &d_host_vector);
