#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cub/warp/warp_scan.cuh>

#include "../../util_device.cuh"
#include "../../graph-utils/load_balanced_search.cuh"

#define VERTEX_FRONTIER 100000

void load_balance_search(int num_edges, const std::vector<int> &scanned_edges, std::vector<int> &result)
{
        int ai = 0, bi = 0;
        while(ai < num_edges || bi < scanned_edges.size())
        {
                bool p;
                if(bi >= scanned_edges.size()) p = true;
                else if(ai >= num_edges) p = false;
                else p = ai < scanned_edges[bi]; // aKey < bKey is upper-bound condition

                if(p) result[ai++] = bi-1; //subtract 1 from the upper-bound
                else ++bi;
        }
}

int main()
{
        std::vector<int> counts(VERTEX_FRONTIER), counts_scan(VERTEX_FRONTIER), sources, lbs;
        for(unsigned i=0; i<counts.size(); i++)
        {
                counts[i] = rand() % 100000; //0 through (k-1) work-items
        }
        /*std::cout << "Number of work items: " << std::endl;
        std::copy(counts.begin(),counts.end(),std::ostream_iterator<int>(std::cout," "));
        std::cout << std::endl;*/

        thrust::exclusive_scan(counts.begin(),counts.end(),counts_scan.begin());
        /*std::cout << "Scanned work items: " << std::endl;
        std::copy(counts_scan.begin(),counts_scan.end(),std::ostream_iterator<int>(std::cout," "));
        std::cout << std::endl;*/

        int edges = counts_scan[counts_scan.size()-1]+counts[counts.size()-1];
        std::cout << "Number of edges to traverse: " << edges << std::endl;
        sources.resize(edges);
        lbs.resize(edges);

        load_balance_search(edges,counts_scan,lbs);
        /*std::cout << "Edges to be traversed: " << std::endl;
        for(unsigned i=0; i<lbs.size(); i++)
        {
                std::cout << "(" << lbs[i] << "," << i - counts_scan[lbs[i]] << ")" << " ";
        }
        std::cout << std::endl;*/

        std::cout << std::endl << "Repeating on the GPU: " << std::endl;
        thrust::device_vector<int> counts_d = counts;
        thrust::device_vector<int> counts_scan_d(VERTEX_FRONTIER,0);
        thrust::device_vector<int> result_d(edges); //Have to assume O(m) space here for a graph
        thrust::device_vector<int> edges_d(1,0);

	cudaEvent_t start_event, end_event;
	start_clock(start_event,end_event);
        extract_edges_block<<<1,BLOCK_SIZE>>>(VERTEX_FRONTIER,thrust::raw_pointer_cast(counts_d.data()),thrust::raw_pointer_cast(counts_scan_d.data()),thrust::raw_pointer_cast(result_d.data()),thrust::raw_pointer_cast(edges_d.data()));
	checkCudaErrors(cudaPeekAtLastError());
	float time = end_clock(start_event,end_event);
        std::cout << "Number of edges to traverse: " << edges_d[0] << std::endl;

        thrust::host_vector<int> result_h = result_d;
        thrust::host_vector<int> counts_scan_h = counts_scan_d;

        /*std::cout << "Edges to be traversed: " << std::endl;
        for(unsigned i=0; i<result_h.size(); i++)
        {
                std::cout << "(" << result_h[i] << "," << i - counts_scan_h[result_h[i]] << ")" << " ";
        }
        std::cout << std::endl;*/

	std::cout << std::endl;

	if(!thrust::equal(counts_scan.begin(),counts_scan.end(),counts_scan_h.begin()))
	{
		std::cout << "Scan failed." << std::endl;
		//thrust::copy(counts_scan_h.begin(),counts_scan_h.end(),std::ostream_iterator<int>(std::cout," " ));
		//std::cout << std::endl;
	}
	else
	{
		std::cout << "Scan passed." << std::endl;
	}
	thrust::equal(lbs.begin(),lbs.end(),result_h.begin()) ? std::cout << "Test passed." : std::cout << "Test failed.";
	std::cout << std::endl;
	std::cout << "Time for Load-Balancing Search: " << std::setprecision(9) << time << " s" << std::endl;

	int64_t bytes = sizeof(int)*edges_d[0]; //The edges themselves are ints but there could be more than 2^31 of them for measurement/testing purposes
	double bandwidth = bytes/time;
	std::cout << "Memory Bandwidth: " << bandwidth/(1e9) << " GB/s" << std::endl;

        return 0;
}
