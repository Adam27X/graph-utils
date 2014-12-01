#pragma once

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <fstream>
#include <cstdlib>

class host_graph
{
public:
	void print_offset_array();
	void print_edge_array();
	void print_from_array();
	void print_adjacency_list();

	//Hybrid CSR/COO representation
	std::vector<int> C; //Array of edges
	std::vector<int> R; //Array of offsets
	std::vector<int> F; //Array of where edges originate from
	int n; //Number of vertices
	int m; //Number of edges

	//Graph attributes
	bool directed;
};

host_graph parse(char *file);
host_graph parse_metis(char *file);
host_graph parse_snap(char *file);
