#pragma once

#include <iostream>
#include <vector>
#include <cstdio>

#include "../shuffle_based.cuh"
#include "../common.cuh"
#include "../../../parse.h"
#include "../../../util_device.cuh"
#include "../../../device_graph.h"

int diameter_sampling_setup(const device_graph &g, int start, int end);
