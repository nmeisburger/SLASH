#include "flashControl.h"

flashControl::flashControl(LSHReservoirSampler *reservoir, CMS *cms, int myRank, int worldSize,
                           unsigned int numDataVectors, unsigned int numQueryVectors,
                           unsigned int dimension, unsigned int numTables,
                           unsigned int reservoirSize) {

    // Core Params
    _myReservoir = reservoir;
    _mySketch = cms;
    _myRank = myRank;
    _worldSize = worldSize;
    _numQueryVectors = numQueryVectors;
    _dimension = dimension;
    _numTables = numTables;
    _reservoirSize = reservoirSize;

    myDataIndices = (unsigned int *)malloc(40);
    myDataVals = (float *)malloc(40);
    myDataMarkers = (unsigned int *)malloc(40);

    _dataVectorOffsets = new unsigned int[_worldSize];
    _dataVectorCts = new unsigned int[_worldSize]();

    _queryVectorOffsets = new int[_worldSize];
    _queryVectorCts = new int[_worldSize]();
    _queryOffsets = new int[_worldSize];
    _queryCts = new int[_worldSize];

    _hashCts = new int[_worldSize];
    _hashOffsets = new int[_worldSize];

    unsigned int queryPartitionSize = std::floor((float)_numQueryVectors / (float)_worldSize);
    unsigned int queryPartitionRemainder = _numQueryVectors % _worldSize;

    for (int j = 0; j < _worldSize; j++) {
        _queryVectorCts[j] = queryPartitionSize;
        if (j < queryPartitionRemainder) {
            _queryVectorCts[j]++;
        }
        _hashCts[j] = _queryVectorCts[j] * _numTables;
    }

    _queryVectorOffsets[0] = 0;
    _hashOffsets[0] = 0;
    for (int n = 1; n < _worldSize; n++) {
        _queryVectorOffsets[n] = std::min(_queryVectorOffsets[n - 1] + _queryVectorCts[n - 1],
                                          (int)_numQueryVectors - 1); // Overflow prevention
        _hashOffsets[n] = _hashOffsets[n - 1] + _hashCts[n - 1];
    }

    _allQueryHashes = new unsigned int[_numQueryVectors * _numTables];

    _myQueryVectorsCt = _queryVectorCts[_myRank];
    _myHashCt = _hashCts[_myRank];

    _queryIndices = new unsigned int[(unsigned)(_numQueryVectors * _dimension)];
    _queryVals = new float[(unsigned)(_numQueryVectors * _dimension)];
    _queryMarkers = new unsigned int[(unsigned)(_numQueryVectors + 1)];

    std::cout << "FLASH Controller Initialized in Node " << _myRank << std::endl;
}

flashControl::~flashControl() {
    delete[] _dataVectorOffsets;
    delete[] _dataVectorCts;

    delete[] _queryVectorOffsets;
    delete[] _queryVectorCts;
    delete[] _queryOffsets;
    delete[] _queryCts;

    delete[] _myQueryIndices;
    delete[] _myQueryVals;
    delete[] _myQueryMarkers;

    delete[] _hashCts;
    delete[] _hashOffsets;

    if (_myRank == 0) {
        delete[] _queryIndices;
        delete[] _queryVals;
        delete[] _queryMarkers;
    }

    delete[] _allQueryHashes;

    free(myDataIndices);
    free(myDataVals);
    free(myDataMarkers);
}
