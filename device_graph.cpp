#include "device_graph.h"

device_graph::device_graph(const host_graph &g)
{
	//Transfer device-side members to the GPU
	C = g.C;
	R = g.R;
	F = g.F;

	//Copy host-side members directly
	n = g.n;
	m = g.m;
	directed = g.directed;
}
