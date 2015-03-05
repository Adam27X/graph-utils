#include <iostream>
#include <queue>
#include <stack>
#include <vector>

#include "../../parse.h"
#include "../../util.h"
#include "../../device_graph.h"
#include "../../graph-utils/multi_search/betweenness_centrality/betweenness_centrality.cuh"

void sequential(host_graph &g_h, int source, std::vector<int> &expected, std::vector<float> &delta)
{
	std::queue<int> Q;
	expected.assign(g_h.n,INT_MAX);
	expected[source] = 0;
	std::vector<unsigned long long> paths(g_h.n,0);
	paths[source] = 1;
	delta.assign(g_h.n,0.0f);
	Q.push(source);
	std::stack<int> S;

	while(!Q.empty())
	{
		int v = Q.front();
		Q.pop();
		S.push(v);

		for(int k=g_h.R[v]; k<g_h.R[v+1]; k++)
		{
			int w = g_h.C[k];
			if(expected[w] == INT_MAX)
			{
				expected[w] = expected[v] + 1;
				Q.push(w);
			}
			if(expected[w] == expected[v] + 1)
			{
				paths[w] += paths[v];
			}
		}
	}

	while(!S.empty())
	{
		int w = S.top();
		S.pop();
		for(int k=g_h.R[w]; k<g_h.R[w+1]; k++)
		{
			int v = g_h.C[k];
			if((expected[v] == (expected[w]-1)) && (v != source))
			{
				delta[v] += (paths[v]/(float)paths[w])*(1.0f+delta[w]);
			}
		}
	}

}

void verify_delta(host_graph &g_h, std::vector< std::vector<float> > &result, int start, int end)
{
	//Obtain sequential result
	const int number_of_rows = result.size(); //Number of SMs on the GPU used for computation
	size_t sources_to_store = (g_h.n < number_of_rows) ? g_h.n - start : number_of_rows;
	std::vector< std::vector<int> > expected(sources_to_store);
	std::vector< std::vector<float> > delta(sources_to_store); //double?
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
		sequential(g_h,sources[i],expected[i],delta[i]);
	}

	//bool match = true;
	int wrong_source;
	//int wrong_dest;
	int wrong_source_index;
	double rms_error = 0.0f;
	double max_error = 0.0f;

        for(unsigned j=0; j<sources_to_store; j++)
        {
                for(int i=0; i<g_h.n; i++)
                {
			double current_error = abs(delta[j][i] - result[j][i]);
			rms_error += current_error*current_error;
			if(current_error > max_error)
			{
				max_error = current_error;
				wrong_source_index = j;
				wrong_source = sources[j];
			}
                }
        }
	rms_error = rms_error/(float)g_h.n;
	rms_error = sqrt(rms_error);
	std::cout << "RMS Error: " << rms_error << std::endl;
	std::cout << "Maximum Error: " << max_error << std::endl;

        if(max_error > 1.0f)
        {
		std::cout << "Source with maximum error: " << wrong_source << std::endl;
                for(int i=0; i<g_h.n; i++)
                {
                        if(i == 0)
                        {
                                std::cout << "Expected = [" << delta[wrong_source_index][i];
                        }
                        else
                        {
                                std::cout << "," << delta[wrong_source_index][i];
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
        }
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
	if(op.approx)
	{
		end = (op.approx > g_h.n) ? g_h.n : op.approx;
	}
	else
	{
		end = (1024 > g_h.n) ? g_h.n : g_h.n; //Some multiple of the number of SMs for now
	}
	std::cout << "Number of source vertices traversed: " << end-start << std::endl;

	std::vector< std::vector<float> > result;	
	betweenness_centrality_setup(g_d,start,end,result);
	if(op.verify)
	{
		verify_delta(g_h,result,start,end);
	}

	return 0;
}

