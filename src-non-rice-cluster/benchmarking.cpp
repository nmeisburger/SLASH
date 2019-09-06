#include <cmath>

#include "LSH.h"
#include "LSHReservoirSampler.h"
#include "CMS.h"
#include "dataset.h"
#include "flashControl.h"
#include "misc.h"
#include "evaluate.h"
#include "indexing.h"
#include "omp.h"
#include "benchmarking.h"
#include "MatMul.h"
#include "mpi.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include "FrequentItems.h"

#define TOPK_BENCHMARK

void controlTest()
{

	MPI_Init(0, 0);

	int myRank, worldSize;

	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

	CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

	LSHReservoirSampler *reservoir = new LSHReservoirSampler(lsh, NUM_HASHES, NUM_TABLES, RESERVOIR_SIZE, DIMENSION,
															 RANGE_ROW_U, NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES, ALLOC_FRACTION, myRank, worldSize);

	flashControl control(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
						 NUM_TABLES, QUERY_PROBES, RESERVOIR_SIZE);

	control.readData("placeholder_txt", 8, 5);

	control.showPartitions();

	control.allocateData();
	control.allocateQuery();

	control.add(1, 1);

	control.hashQuery();

	unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];

	control.extractReservoirs(TOPK, outputs);

	printf("TOP K..\n");
	for (int i = 0; i < TOPK * NUM_QUERY_VECTORS; i++)
	{
		printf("\t%d", outputs[i]);
	}
	printf("\n");

	delete reservoir;
	delete lsh;

	delete[] outputs;

	MPI_Finalize();
}

void webspamTest()
{
	int provided;
	MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);

	int myRank, worldSize;

	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	std::cout << "Hashing Probes: " << HASHING_PROBES << " Query Probes: " << QUERY_PROBES << std::endl;

	LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

	CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

	LSHReservoirSampler *reservoir = new LSHReservoirSampler(lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION,
															 RANGE_ROW_U, NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES, ALLOC_FRACTION, myRank, worldSize);

	flashControl *control = new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
											 NUM_TABLES, QUERY_PROBES, RESERVOIR_SIZE);

	if (myRank == 0) {
		std::cout << "\nReading Data Node 0..." << std::endl;
	}
	auto start = std::chrono::system_clock::now();
	control->readData(BASEFILE, NUM_DATA_VECTORS + NUM_QUERY_VECTORS, DIMENSION);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	if (myRank == 0) {
		std::cout << "Data Read Node 0: " << elapsed.count() << " Seconds\n" << std::endl;
	}

	unsigned int *gtruth_indice = new unsigned int[NUM_QUERY_VECTORS * AVAILABLE_TOPK];
	float *gtruth_dist = new float[NUM_QUERY_VECTORS * AVAILABLE_TOPK];
	if (myRank == 0) {
		std::cout << "Reading Groundtruth Node 0..." << std::endl;	
		start = std::chrono::system_clock::now();
		readGroundTruthInt(GTRUTHINDICE, NUM_QUERY_VECTORS, AVAILABLE_TOPK, gtruth_indice);
		readGroundTruthFloat(GTRUTHDIST, NUM_QUERY_VECTORS, AVAILABLE_TOPK, gtruth_dist);
		end = std::chrono::system_clock::now();
		elapsed = end - start;
		std::cout << "Groundtruth Read Node 0: " << elapsed.count() << " Seconds\n" << std::endl;
	}

	control->showPartitions();
	control->allocateData();
	control->allocateQuery();

	std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->add(NUM_BATCHES, BATCH_PRINT);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->hashQuery();
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
	start = std::chrono::system_clock::now();
	std::cout << "Extracting Top K (CMS) Node " << myRank << "..." << std::endl;
	control->extractReservoirsCMS(TOPK, outputs, 0);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

#ifdef TOPK_BENCHMARK
	unsigned int *outputs2 = new unsigned int[TOPK * NUM_QUERY_VECTORS];
	start = std::chrono::system_clock::now();
	std::cout << "Extracting Top K (Standard) Node " << myRank << "..." << std::endl;
	control->extractReservoirs(TOPK, outputs2);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;
#endif

	if (myRank == 0) {
		const int nCnt = 10;
		int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};
		const int gstdCnt = 8;
		float gstdVec[gstdCnt] = {0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50};
		const int tstdCnt = 10;
		int tstdVec[tstdCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

		std::cout << "\n\n================================\nTOP K CMS\n"
				  << std::endl;

		similarityMetric(control->_sparseIndices, control->_sparseVals, control->_sparseMarkers,
						 control->_sparseIndices, control->_sparseVals, control->_sparseMarkers, outputs, gtruth_dist,
						 NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
		std::cout << "Similarity Metric Computed" << std::endl;
		similarityOfData(gtruth_dist, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
		std::cout << "Similarity of Data Computed" << std::endl;

		for (int i = 0; i < NUM_QUERY_VECTORS * TOPK; i++) {
			outputs[i] -= NUM_QUERY_VECTORS;
		}

		evaluate(outputs, NUM_QUERY_VECTORS, TOPK, gtruth_indice, gtruth_dist, AVAILABLE_TOPK, gstdVec, gstdCnt, tstdVec, tstdCnt, nList, nCnt);
		std::cout << "Evaluation Complete" << std::endl;

#ifdef TOPK_BENCHMARK
		std::cout << "\n\n================================\nTOP K STANDARD\n"
				  << std::endl;

		similarityMetric(control->_sparseIndices, control->_sparseVals, control->_sparseMarkers,
						 control->_sparseIndices, control->_sparseVals, control->_sparseMarkers, outputs2, gtruth_dist,
						 NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
		std::cout << "Similarity Metric Computed" << std::endl;
		similarityOfData(gtruth_dist, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
		std::cout << "Similarity of Data Computed" << std::endl;

		for (int i = 0; i < NUM_QUERY_VECTORS * TOPK; i++) {
			outputs2[i] -= NUM_QUERY_VECTORS;
		}

		evaluate(outputs2, NUM_QUERY_VECTORS, TOPK, gtruth_indice, gtruth_dist, AVAILABLE_TOPK, gstdVec, gstdCnt, tstdVec, tstdCnt, nList, nCnt);
		std::cout << "Evaluation Complete" << std::endl;
#endif
	}

	delete[] outputs;
#ifdef TOPK_BENCHMARK
	delete[] outputs2;
#endif

	if (myRank == 0) {
		delete[] gtruth_dist;
		delete[] gtruth_indice;
	}

	delete control;
	delete reservoir;
	delete lsh;
	delete cms;

	MPI_Finalize();
}