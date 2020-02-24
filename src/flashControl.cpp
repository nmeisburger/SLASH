#include "flashControl.h"

void flashControl::allocateQuery(std::string filename) {

    if (_myRank == 0) {
        _queryIndices = new unsigned int[(unsigned)(_numQueryVectors * _dimension)];
        _queryVals = new float[(unsigned)(_numQueryVectors * _dimension)];
        _queryMarkers = new unsigned int[(unsigned)(_numQueryVectors + 1)];
        readSparse(filename, 0, _numQueryVectors, _queryIndices, _queryVals, _queryMarkers,
                   (unsigned)(_numQueryVectors * _dimension));

        for (int n = 0; n < _worldSize; n++) {
            _queryOffsets[n] = _queryMarkers[_queryVectorOffsets[n]];
            _queryCts[n] =
                _queryMarkers[_queryVectorOffsets[n] + _queryVectorCts[n]] - _queryOffsets[n];
        }
    }

    MPI_Bcast(_queryOffsets, _worldSize, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(_queryCts, _worldSize, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    _myQueryVectorsLen = _queryCts[_myRank];

    _myQueryIndices = new unsigned int[_myQueryVectorsLen];
    _myQueryVals = new float[_myQueryVectorsLen];
    _myQueryMarkers = new unsigned int[_myQueryVectorsCt + 1];

    int *tempQueryMarkerCts = new int[_worldSize];
    for (int n = 0; n < _worldSize; n++) {
        tempQueryMarkerCts[n] =
            _queryVectorCts[n] + 1; // To account for extra element at the end of each marker array
    }

    MPI_Scatterv(_queryIndices, _queryCts, _queryOffsets, MPI_UNSIGNED, _myQueryIndices,
                 _myQueryVectorsLen, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_queryVals, _queryCts, _queryOffsets, MPI_FLOAT, _myQueryVals, _myQueryVectorsLen,
                 MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_queryMarkers, tempQueryMarkerCts, _queryVectorOffsets, MPI_UNSIGNED,
                 _myQueryMarkers, _myQueryVectorsCt + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int myQueryOffset = _queryOffsets[_myRank];
    for (size_t i = 0; i < _myQueryVectorsCt + 1; i++) {
        _myQueryMarkers[i] -= myQueryOffset;
    }
    delete[] tempQueryMarkerCts;
}

void flashControl::add(std::string filename, unsigned int numDataVectors, unsigned int offset,
                       unsigned int numBatches, unsigned int batchPrint) {

    _myReservoir->resetSequentialKernalID();

    unsigned int dataPartitionSize = numDataVectors / _worldSize;
    unsigned int dataPartitionRemainder = numDataVectors % _worldSize;

    for (int i = 0; i < _worldSize; i++) {
        _dataVectorCts[i] = dataPartitionSize;
        if (i < dataPartitionRemainder) {
            _dataVectorCts[i]++;
        }
    }

    _dataVectorOffsets[0] = offset;

    for (int n = 1; n < _worldSize; n++) {
        _dataVectorOffsets[n] = std::min(_dataVectorOffsets[n - 1] + _dataVectorCts[n - 1],
                                         numDataVectors + offset - 1); // Overflow prevention
    }

    unsigned int myNumDataVectors = _dataVectorCts[_myRank];
    unsigned int myDataVectorOffset = _dataVectorOffsets[_myRank];

    unsigned int *myDataIndices = new unsigned int[myNumDataVectors * _dimension];
    float *myDataVals = new float[myNumDataVectors * _dimension];
    unsigned int *myDataMarkers = new unsigned int[myNumDataVectors + 1];

    readSparse(filename, myDataVectorOffset, myNumDataVectors, myDataIndices, myDataVals,
               myDataMarkers, myNumDataVectors * _dimension);

    unsigned int batchSize = myNumDataVectors / numBatches;
    for (unsigned int batch = 0; batch < numBatches; batch++) {
        _myReservoir->add(batchSize, myDataIndices, myDataVals, myDataMarkers + batch * batchSize,
                          myDataVectorOffset);
        if (batch % batchPrint == 0) {
            _myReservoir->checkTableMemLoad();
        }
    }

    delete[] myDataIndices;
    delete[] myDataVals;
    delete[] myDataMarkers;
}

void flashControl::hashQuery() {

    unsigned int *myPartitionHashes = new unsigned int[_myHashCt];

    _myReservoir->getQueryHash(_myQueryVectorsCt, _myHashCt, _myQueryIndices, _myQueryVals,
                               _myQueryMarkers, myPartitionHashes);

    unsigned int *queryHashBuffer = new unsigned int[_numQueryVectors * _numTables];

    MPI_Allgatherv(myPartitionHashes, _myHashCt, MPI_UNSIGNED, queryHashBuffer, _hashCts,
                   _hashOffsets, MPI_UNSIGNED, MPI_COMM_WORLD);

    unsigned int len;

    unsigned int *old;
    unsigned int *fin;

#pragma omp parallel for default(none)                                                             \
    shared(queryHashBuffer, _allQueryHashes, _hashOffsets, _numTables) private(len, old, fin)
    for (size_t partition = 0; partition < _worldSize; partition++) {
        len = _queryVectorCts[partition];
        for (size_t tb = 0; tb < _numTables; tb++) {
            old = queryHashBuffer + _hashOffsets[partition] + tb * len;
            fin = _allQueryHashes + tb * _numQueryVectors + (_hashOffsets[partition] / _numTables);
            for (size_t l = 0; l < len; l++) {
                fin[l] = old[l];
            }
        }
    }

    delete[] queryHashBuffer;
    delete[] myPartitionHashes;
}

void flashControl::topKBruteForceAggretation(unsigned int topK, unsigned int *outputs) {
    size_t segmentSize = _numTables * _reservoirSize;
    unsigned int *allReservoirsExtracted = new unsigned int[segmentSize * (long)_numQueryVectors];
    _myReservoir->extractReservoirs(_numQueryVectors, segmentSize, allReservoirsExtracted,
                                    _allQueryHashes);

    unsigned int *allReservoirsAllNodes;
    if (_myRank == 0) {
        allReservoirsAllNodes =
            new unsigned int[segmentSize * (long)_numQueryVectors * (long)_worldSize];
    }
    MPI_Gather(allReservoirsExtracted, segmentSize * _numQueryVectors, MPI_UNSIGNED,
               allReservoirsAllNodes, segmentSize * _numQueryVectors, MPI_UNSIGNED, 0,
               MPI_COMM_WORLD);

    if (_myRank == 0) {
        unsigned int *allReservoirsAllNodesOrdered =
            new unsigned int[segmentSize * _numQueryVectors * _worldSize];

        unsigned int queryBlockSize = _worldSize * segmentSize;
        unsigned int *old;
        unsigned int *final;

#pragma omp parallel for default(none)                                                             \
    shared(allReservoirsAllNodes, allReservoirsAllNodesOrdered, queryBlockSize, segmentSize,       \
           _numQueryVectors) private(old, final)
        for (size_t v = 0; v < _numQueryVectors; v++) {
            for (size_t n = 0; n < _worldSize; n++) {
                old =
                    allReservoirsAllNodes + v * segmentSize + n * (_numQueryVectors * segmentSize);
                final = allReservoirsAllNodesOrdered + v * queryBlockSize + n * segmentSize;
                for (size_t i = 0; i < segmentSize; i++) {
                    final[i] = old[i];
                }
            }
        }

        delete[] allReservoirsAllNodes;

#pragma omp parallel for default(none) shared(allReservoirsAllNodesOrdered, queryBlockSize)
        for (size_t v = 0; v < _numQueryVectors; v++) {
            std::sort(allReservoirsAllNodesOrdered + v * queryBlockSize,
                      allReservoirsAllNodesOrdered + (v + 1) * queryBlockSize);
        }

        VectorFrequency *vectorCnts =
            new VectorFrequency[segmentSize * _numQueryVectors * _worldSize];

#pragma omp parallel for default(none)                                                             \
    shared(allReservoirsAllNodesOrdered, vectorCnts, queryBlockSize, outputs, topK)
        for (size_t v = 0; v < _numQueryVectors; v++) {
            unsigned int uniqueVectors = 0;
            unsigned int current = allReservoirsAllNodesOrdered[0];
            unsigned int count = 1;
            for (size_t i = 1; i < queryBlockSize; i++) {
                if (allReservoirsAllNodesOrdered[i + v * queryBlockSize] == current) {
                    count++;
                } else {
                    vectorCnts[uniqueVectors + v * queryBlockSize].vector = current;
                    vectorCnts[uniqueVectors + v * queryBlockSize].count = count;
                    current = allReservoirsAllNodesOrdered[i + v * queryBlockSize];
                    count = 1;
                    uniqueVectors++;
                }
            }
            vectorCnts[uniqueVectors + v * queryBlockSize].vector = current;
            vectorCnts[uniqueVectors + v * queryBlockSize].count = count;
            uniqueVectors++;
            for (; uniqueVectors < queryBlockSize; uniqueVectors++) {
                vectorCnts[uniqueVectors + v * queryBlockSize].count = -1;
            }
            std::sort(
                vectorCnts + v * queryBlockSize, vectorCnts + (v + 1) * queryBlockSize,
                [&vectorCnts](VectorFrequency a, VectorFrequency b) { return a.count > b.count; });
            int s = 0;
            if (vectorCnts[queryBlockSize * v].vector == 0)
                s++;
            for (size_t k = 0; k < topK; k++) {
                outputs[k + topK * v] = vectorCnts[s + k + v * queryBlockSize].vector;
            }
        }
    }
}

void flashControl::topKCMSAggregationTree(unsigned int topK, unsigned int *outputs,
                                          unsigned int threshold) {
    unsigned int segmentSize = _numTables * _reservoirSize;
    unsigned int *allReservoirsExtracted = new unsigned int[segmentSize * _numQueryVectors];
    _myReservoir->extractReservoirs(_numQueryVectors, segmentSize, allReservoirsExtracted,
                                    _allQueryHashes);

    _mySketch->add(allReservoirsExtracted, segmentSize);

    _mySketch->aggregateSketchesTree();

    if (_myRank == 0) {
        _mySketch->topK(topK, outputs, threshold);
    }

    delete[] allReservoirsExtracted;
}

void flashControl::topKCMSAggregationLinear(unsigned int topK, unsigned int *outputs,
                                            unsigned int threshold) {
    unsigned int segmentSize = _numTables * _reservoirSize;
    unsigned int *allReservoirsExtracted = new unsigned int[segmentSize * _numQueryVectors];
    _myReservoir->extractReservoirs(_numQueryVectors, segmentSize, allReservoirsExtracted,
                                    _allQueryHashes);

    _mySketch->add(allReservoirsExtracted, segmentSize);

    _mySketch->aggregateSketches();

    if (_myRank == 0) {
        _mySketch->topK(topK, outputs, threshold);
    }

    delete[] allReservoirsExtracted;
}

void flashControl::printTables() {
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            _myReservoir->tableContents();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void flashControl::showPartitions() {
    printf("[Status Rank %d]:\n\tQuery Vector Range: [%d, "
           "%d)\n\tQuery Range: [%d, %d)\n\n",
           _myRank, _queryVectorOffsets[_myRank], _queryVectorOffsets[_myRank] + _myQueryVectorsCt,
           _queryOffsets[_myRank], _queryOffsets[_myRank] + _myQueryVectorsLen);
}

void flashControl::checkQueryHashes() {
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            int hashOffset = _hashOffsets[_myRank];
            printf("Query Hashes Node %d\n", n);
            for (size_t h = 0; h < _myHashCt; h++) {
                printf("\tHash %zu: %d\n", hashOffset + h, _allQueryHashes[hashOffset + h]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (_myRank == 0) {
        printf("\n\nCombined Query Hashes\n");
        for (int h = 0; h < _numQueryVectors * _numTables; h++) {
            printf("\tHash %d: %d\n", h, _allQueryHashes[h]);
        }
    }
}
