#include <iostream>

#include "../../parse.h"
#include "../../util.h"

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);

	host_graph g = parse(op);

	std::cout << "Graph parsed." << std::endl;

	/*g.print_offset_array();
	g.print_edge_array();
	g.print_from_array();
	g.print_adjacency_list();*/
	std::cout << "Number of vertices with no outdegree: " << g.count_degree_zero_vertices() << std::endl;

	std::cout << "Adjacency list of vertex 0: " << std::endl;
	for(int i=g.R[0]; i<g.R[1]; i++)
	{
		std::cout << g.C[i] << " ";
	}
	std::cout << std::endl;

	return 0;
}
