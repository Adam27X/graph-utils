#include <iostream>
#include <queue>

#include "../../parse.h"
#include "../../util.h"
#include "../../device_graph.h"
#include "../../graph-utils/multi_search/linear_with_atomics.cuh"

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

bool verify_linear(host_graph &g_h, std::vector< std::vector<int> > &result)
{
	//Obtain sequential result
	std::vector< std::vector<int> > expected(5);
	for(int i=0; i<5; i++)
	{
		sequential(g_h,i,expected[i]);
	}

	bool match = true;
	for(int j=0; j<5; j++)
	{
		for(int i=0; i<g_h.n; i++)
		{
			if(expected[j][i] != result[j][i])
			{
				match = false;
				std::cout << "Mismatch in results for vertex " << i << ". Expected: " << expected[j][i] << ". Actual: " << result[j][i] << "." << std::endl;
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
	start = 0;
	end = (56 > g_h.n) ? 56 : g_h.n; //Some multiple of the number of SMs for now

	std::vector< std::vector<int> > result = multi_search_linear_atomics_setup(g_d,start,end);

	bool pass = verify_linear(g_h,result);
	if(pass)
	{
		std::cout << "Test passed." << std::endl;
	}

	return 0;
}

