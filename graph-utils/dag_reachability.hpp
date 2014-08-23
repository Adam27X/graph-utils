#pragma once

#include <iostream>
#include <set>

#include "reverse.hpp"

//This function keeps track of recursive calls to report a cycle in the case that one is found. If this functionality impacts performance it can be removed in favor a function that assumes a DAG input.
void find_reach(graph &g, int i, std::vector< std::set<int> > &reach, std::set<int> &chain)
{
	//Base case: If a vertex has no outdegree then it cannot reach anything.
	if((g.R[i+1]-g.R[i]) == 0)
	{
		return;
	}
	else //Otherwise, recurse
	{
		std::set<int> tmp;
		for(int j=g.R[i]; j<g.R[i+1]; j++)
		{
			int w = g.C[j];
			tmp.insert(w);
			if(reach[w].empty())
			{
				if((chain.find(w) != chain.end()) && ((g.R[w+1]-g.R[w]) != 0))
				{
					std::cout << "Cycle found!" << std::endl;
					std::cout << "Element to be inserted: " << w << std::endl;
					std::cout << "Chain: ";
					for(auto i=chain.begin(),e=chain.end(); i!=e; ++i)
					{
						std::cout << *i << " ";
					}
					std::cout << std::endl;
					exit(0);
				}
				else
				{
					chain.insert(w);
					find_reach(g,w,reach,chain);
					tmp.insert(reach[w].begin(),reach[w].end());
				}
			}
			else
			{
				tmp.insert(reach[w].begin(),reach[w].end());
			}
		}
		reach[i] = tmp;
	}
}

//Given a graph, return a vector of sets representing reachable vertices for each node
//The graph MUST be a directed acyclic graph for this method to work properly
std::vector< std::set<int> > get_reachability(graph &g)
{
	//First, we need to obtain the reverse graph
	graph g_rev = reverse(g);

	//Collect vertices with indegree 0
	std::vector<int> roots;
	for(int i=0; i<g.n; i++)
	{
		if((g_rev.R[i+1]-g_rev.R[i]) == 0)
		{
			roots.push_back(i);
		}
	}

	//Find reachability for each root
	std::vector< std::set<int> > reach(g.n);
	for(auto i=roots.begin(),e=roots.end(); i!=e; ++i)
	{
		std::set<int> chain_begin(i,i+1);
		find_reach(g,*i,reach,chain_begin);
	}

	return reach;
}
