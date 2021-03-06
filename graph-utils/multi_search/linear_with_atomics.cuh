#pragma once

#include <vector>
#include <cuda.h>
#include <cstdio>
#include <iostream>

#include "../../parse.h"
#include "../../device_graph.h"
#include "../../util_device.cuh"

//TODO: Use some "flexible" CUDA graph data structure
__global__ void multi_search_linear_atomics(const int *R, const int *C, const int n, int *d, size_t pitch_d, int *Q, size_t pitch_Q, int *Q2, size_t pitch_Q2, const int start, const int end);
std::vector< std::vector<int> > multi_search_linear_atomics_setup(const device_graph &g, int start, int end);
