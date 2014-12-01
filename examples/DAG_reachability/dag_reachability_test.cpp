#include <iostream>

#include "../../parse.h"
#include "../../util.h"
#include "../../graph-utils/dag_reachability.h"

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);

	host_graph g = parse(op.infile);

	host_graph g_rev = reverse(g);

	std::vector< std::set<int> > reach = get_reachability_dag(g);

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
