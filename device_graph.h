#pragma once
#include <thrust/device_vector.h>
#include "parse.h"

//Device side copy of a graph. Once it has been constructed, it should function on its own without a corresponding host graph.
//TODO: Functions for smoothly copying/to from a host graph, helper routines, etc.
class device_graph
{
public: 
	//Copy constructor from host_graph
	device_graph(const host_graph& g);

        //Hybrid CSR/COO representation
        thrust::device_vector<int> C; //Array of edges
        thrust::device_vector<int> R; //Array of offsets
        thrust::device_vector<int> F; //Array of where edges originate from
        int n; //Number of vertices
        int m; //Number of edges

        //Graph attributes
        bool directed;
};
