#include <iostream>
#include <queue>

#include "../../parse.h"
#include "../../util.h"
#include "../../device_graph.h"
#include "../../graph-utils/multi_search/linear_with_atomics.cuh"
#include "../../graph-utils/multi_search/edge_parallel.cuh"
#include "../../graph-utils/multi_search/warp_based.cuh"
#include "../../graph-utils/multi_search/scan_based.cuh"

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
	int wrong_source;
	int wrong_dest;
	for(int j=start; j<start+5; j++)
	{
		for(int i=0; i<g_h.n; i++)
		{
			if(expected[j-start][i] != result[j-start][i])
			{
				match = false;
				wrong_source = j;
				wrong_dest = i;
				std::cout << "Mismatch for source " << wrong_source << " and dest " << wrong_dest << std::endl;
				std::cout << "Expected distance: " << expected[j-start][i] << std::endl;
				std::cout << "Actual distance: " << result[j-start][i] << std::endl;
				break;
			}
		}
		if(match == false)
		{
			break;
		}
	}

	/*if(match == false)
	{
		std::cout << "Mismatch for source " << wrong_source << std::endl;
		for(int i=0; i<g_h.n; i++)
		{
			if(i == 0)
			{
				std::cout << "Expected = [" << expected[wrong_source-start][i];
			}
			else
			{
				std::cout << "," << expected[wrong_source-start][i];
			}
		}
		std::cout << "]" << std::endl;
		
		for(int i=0; i<g_h.n; i++)
		{
			if(i == 0)
			{
				if(i == wrong_dest)
				{
					std::cout << "Actual = [\033[1;31m" << result[wrong_source-start][i] << "\033[0m";
				}
				else
				{
					std::cout << "Actual = [" << result[wrong_source-start][i];
				}
			}
			else
			{	
				if(i == wrong_dest)
				{
					std::cout << ",\033[1;31m" << result[wrong_source-start][i] << "\033[0m";
				}
				else
				{
					std::cout << "," << result[wrong_source-start][i];
				}
			}
		}
		std::cout << "]" << std::endl;
	}*/

	return match;
}

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);
	
	host_graph g_h = parse(op.infile);
	std::cout << "Number of vertices: " << g_h.n << std::endl;
	std::cout << "Number of (directed) edges: " << g_h.m << std::endl;
	device_graph g_d(g_h);
	int start,end;
	start = 2; //TODO: Use start and end to determine which vertices will be BFS source, but store/verify results for a small subset rather than all of them
	end = (1024 > g_h.n) ? g_h.n : 1024; //Some multiple of the number of SMs for now

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

	result = multi_search_warp_based_setup(g_d,start,end);
	pass = verify_multi_search(g_h,result,start);
	if(pass)
	{
		std::cout << "Warp based: Test passed." << std::endl;
	}

	result = multi_search_scan_based_setup(g_d,start,end);
	pass = verify_multi_search(g_h,result,start);
	if(pass)
	{
		std::cout << "Scan based: Test passed." << std::endl;
	}

	return 0;
}

