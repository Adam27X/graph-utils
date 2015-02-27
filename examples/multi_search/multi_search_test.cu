#include <iostream>
#include <queue>
#include <bitset>

#include "../../parse.h"
#include "../../util.h"
#include "../../device_graph.h"
#include "../../graph-utils/multi_search/linear_with_atomics.cuh"
#include "../../graph-utils/multi_search/edge_parallel.cuh"
#include "../../graph-utils/multi_search/warp_based.cuh"
#include "../../graph-utils/multi_search/scan_based.cuh"
#include "../../graph-utils/multi_search/CTA_warp.cuh"
#include "../../graph-utils/multi_search/shuffle_based.cuh"

//This is essentially the APSP problem (same definition used by Aydin Buluc in PARCO, 2009.
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

bool verify_multi_search(host_graph &g_h, std::vector< std::vector<int> > &result, int start, int end)
{
	//Obtain sequential result
	const int number_of_rows = result.size(); //Number of SMs on the GPU used for computation
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

	for(unsigned i=0; i<sources_to_store; i++)
	{
		sequential(g_h,sources[i],expected[i]);
	}

	bool match = true;
	int wrong_source;
	int wrong_dest;
	int wrong_source_index;

	for(unsigned j=0; j<sources_to_store; j++)
	{
		for(int i=0; i<g_h.n; i++)
		{
			if(expected[j][i] != result[j][i])
			{
				match = false;
				wrong_source = sources[j]; 
				wrong_source_index = j;
				wrong_dest = i;
				std::cout << "Mismatch for source " << wrong_source << " and dest " << wrong_dest << std::endl;
				std::cout << "Expected distance: " << expected[j][i] << std::endl;
				std::cout << "Actual distance: " << result[j][i] << std::endl;
				//break;
			}
		}
		/*if(match == false)
		{
			break;
		}*/
	}

	/*if(match == false)
	{
		for(int i=0; i<g_h.n; i++)
		{
			if(i == 0)
			{
				std::cout << "Expected = [" << expected[wrong_source_index][i];
			}
			else
			{
				std::cout << "," << expected[wrong_source_index][i];
			}
		}
		std::cout << "]" << std::endl;
		
		for(int i=0; i<g_h.n; i++)
		{
			if(i == 0)
			{
				std::cout << "Actual = [" << result[wrong_source_index][i];
			}
			else
			{	
				std::cout << "," << result[wrong_source_index][i];
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

	unsigned algorithm_choice;
	std::cout << "Choose which algorithms to run." << std::endl;
	std::cout << "Add 1 for linear with atomics, 2 for edge_parallel, 4 for warp-based, 8 for scan-based, and 16 for shuffle based." << std::endl;
	std::cin >> algorithm_choice;
	std::bitset<8> alg(algorithm_choice);

	std::vector< std::vector<int> > result;
	bool pass;

	if(alg[0])
	{
		result = multi_search_linear_atomics_setup(g_d,start,end);
		pass = verify_multi_search(g_h,result,start,end);
		if(pass)
		{
			std::cout << "Linear with atomics: Test passed." << std::endl;
		}
	}

	if(alg[1])
	{
		result = multi_search_edge_parallel_setup(g_d,start,end);
		pass = verify_multi_search(g_h,result,start,end);
		if(pass)
		{
			std::cout << "Edge parallel: Test passed." << std::endl;
		}
	}

	if(alg[2])
	{
		result = multi_search_warp_based_setup(g_d,start,end);
		pass = verify_multi_search(g_h,result,start,end);
		if(pass)
		{
			std::cout << "Warp based: Test passed." << std::endl;
		}
	}

	if(alg[3])
	{
		result = multi_search_scan_based_setup(g_d,start,end);
		pass = verify_multi_search(g_h,result,start,end);
		if(pass)
		{
			std::cout << "Scan based: Test passed." << std::endl;
		}
	}
		
	if(alg[4])
	{
		result = multi_search_shuffle_based_setup(g_d,start,end);
		if(op.verify)
		{
			pass = verify_multi_search(g_h,result,start,end);
			if(pass)
			{
				std::cout << "Shuffle based: Test passed." << std::endl;
			}
		}
	}

	/*result = multi_search_CTA_warp_based_setup(g_d,start,end);
	pass = verify_multi_search(g_h,result,start,end);
	if(pass)
	{
		std::cout << "CTA+warp: Test passed." << std::endl;
	}*/

	return 0;
}

