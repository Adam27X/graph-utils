#include <iostream>
#include <queue>
#include <vector>

#include "../../parse.h"
#include "../../util.h"
#include "../../device_graph.h"
#include "../../graph-utils/multi_search/all_pairs_shortest_paths/all_pairs_shortest_paths.cuh"

void sequential(host_graph &g_h, int source, std::vector<int> &expected, int &max)
{
	std::queue<int> Q;
	expected.assign(g_h.n,INT_MAX);
	expected[source] = 0;
	Q.push(source);
	while(!Q.empty())
	{
		int v = Q.front();
		Q.pop();

		for(int k=g_h.R[v]; k<g_h.R[v+1]; k++)
		{
			int w = g_h.C[k];
			if(expected[w] == INT_MAX)
			{
				expected[w] = expected[v] + 1;
				Q.push(w);
			}
		}
	}

}

bool verify_apsp(host_graph &g_h, int result, int start, int end)
{
	//Obtain sequential result
	const int number_of_rows = g_h.n; //Number of SMs on the GPU used for computation
	size_t sources_to_store = (g_h.n < number_of_rows) ? g_h.n - start : number_of_rows;
	std::vector< std::vector<int> > expected(sources_to_store);
	std::vector<int> sources(sources_to_store);
	if(g_h.n < number_of_rows)
	{
		for(unsigned i=0; i<sources_to_store; i++)
		{
			sources[i] = start+i;
		}	
	}
	else
	{
		int earliest_source_val = end - number_of_rows;
		int location_of_earliest_source = (end-start) % number_of_rows;
		for(unsigned i=0; i<sources_to_store; i++)
		{
			int index = (location_of_earliest_source+i)%number_of_rows;
			sources[index] = earliest_source_val+i;
		}	
	}

	int max = 0;
	for(unsigned i=0; i<sources_to_store; i++)
	{
		sequential(g_h,sources[i],expected[i],max);
	}

	return result == max;
}

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);
	
	host_graph g_h = parse(op.infile);
	std::cout << "Number of vertices: " << g_h.n << std::endl;
	std::cout << "Number of (directed) edges: " << g_h.m << std::endl;
	int maxdegree = 0;
	for(int i=0; i<g_h.n; i++)
	{
		if(g_h.R[i+1]-g_h.R[i] > maxdegree)
		{
			maxdegree = g_h.R[i+1]-g_h.R[i];
		}
	}
	std::cout << "Maximum outdegree: " << maxdegree << std::endl << std::endl;

	choose_device(op);
	std::cout << std::endl;

	device_graph g_d(g_h);
	int start,end;
	start = 0; 
	end = (1024 > g_h.n) ? g_h.n : g_h.n; //Some multiple of the number of SMs for now
	
	std::vector< std::vector<unsigned long long> > result = all_pairs_shortest_paths_setup(g_d,start,end);

	return 0;
}

