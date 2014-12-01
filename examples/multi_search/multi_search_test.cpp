#include <iostream>

#include "../../parse.h"
#include "../../util.h"
#include "../../device_graph.h"
#include "../../graph-utils/multi_search/linear_with_atomics.cuh"

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);
	
	host_graph g_h = parse(op.infile);
	device_graph g_d(g_h);

	std::vector< std::vector<int> > result = multi_search_linear_atomics_setup(g_d,0,g_d.n);

	for(int i=0; i<g_d.n; i++)
	{
		std::cout << "BFS results for vertex " << i << ": " << std::endl;
		for(int j=0; j<g_d.n; j++)
		{
			if(j == 0)
			{
				std::cout << "[" << result[i][j];
			}
			else
			{
				std::cout << "," << result[i][j];
			}
		}
		std::cout << "]" << std::endl;
	}

	return 0;
}
