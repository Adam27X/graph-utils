#include <iostream>

#include "../../parse.h"
#include "../../util.h"
#include "../../graph-utils/reverse.hpp"
#include "../../graph-utils/dag_reachability.hpp"

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);

	graph g = parse(op.infile);

	graph g_rev = reverse(g);

	g.print_adjacency_list();
	g_rev.print_adjacency_list();

	std::vector< std::set<int> > reach = get_reachability(g);

	std::cout << "Reachability sets: " << std::endl;
	for(int i=0; i<g.n; i++)
	{
		std::cout << i << " | ";
		for(auto j=reach[i].begin(); j!=reach[i].end(); ++j)
		{
			std::cout << *j << " ";
		} 
		std::cout << std::endl;
	}
	
	return 0;
}
