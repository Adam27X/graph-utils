#include "dag_reachability.h"

//This function assumes a DAG input. A stack overflow can occur otherwise.
void find_reach(graph &g, int i, std::vector< std::set<int> > &reach)
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
				find_reach(g,w,reach);
				//tmp.insert(reach[w].begin(),reach[w].end());
				std::set_union(reach[w].begin(),reach[w].end(),reach[i].begin(),reach[i].end(),std::inserter(tmp,tmp.end()));
			}
			else
			{
				//tmp.insert(reach[w].begin(),reach[w].end());
				std::set_union(reach[w].begin(),reach[w].end(),reach[i].begin(),reach[i].end(),std::inserter(tmp,tmp.end()));
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
		find_reach(g,*i,reach);
	}

	return reach;
}
