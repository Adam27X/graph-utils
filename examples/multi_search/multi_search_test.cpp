#include <iostream>
#include <queue>

#include "../../parse.h"
#include "../../util.h"
#include "../../device_graph.h"
#include "../../graph-utils/multi_search/linear_with_atomics.cuh"
#include "../../graph-utils/multi_search/edge_parallel.cuh"

void sequential(host_graph &g_h, int source, std::vector<int> &expected)
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

bool verify_multi_search(host_graph &g_h, std::vector< std::vector<int> > &result, int start)
{
	//Obtain sequential result
	std::vector< std::vector<int> > expected(5);
	for(int i=start; i<start+5; i++)
	{
		sequential(g_h,i,expected[i-start]);
	}

	bool match = true;
	for(int j=start; j<start+5; j++)
	{
		for(int i=0; i<g_h.n; i++)
		{
			if(expected[j-start][i] != result[j-start][i])
			{
				match = false;
				std::cout << "Mismatch in results for vertex " << i << ". Expected: " << expected[j-start][i] << ". Actual: " << result[j-start][i] << "." << std::endl;
			}
		}
	}

	return match;
}

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);
	
	host_graph g_h = parse(op.infile);
	device_graph g_d(g_h);
	int start,end;
	start = 2;
	end = (56 > g_h.n) ? 56 : g_h.n; //Some multiple of the number of SMs for now

	std::vector< std::vector<int> > result = multi_search_linear_atomics_setup(g_d,start,end);
	bool pass = verify_multi_search(g_h,result,start);
	if(pass)
	{
		std::cout << "Linear with atomics: Test passed." << std::endl;
	}

	result = multi_search_edge_parallel_setup(g_d,start,end);
	pass = verify_multi_search(g_h,result,start);
	if(pass)
	{
		std::cout << "Edge parallel: Test passed." << std::endl;
	}

	return 0;
}

