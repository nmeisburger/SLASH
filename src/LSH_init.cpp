#include "LSH.h"

LSH::LSH(DOPH *hashFamIn, unsigned int numHashPerFamily, unsigned int numHashFamilies,
         unsigned int reservoirSize, unsigned int dimension, unsigned int maxSamples, int myRank,
         int worldSize) {

    _myRank = myRank;
    _worldSize = worldSize;

    initVariables(numHashPerFamily, numHashFamilies, reservoirSize, dimension, maxSamples);

    _hashFamily = hashFamIn;

    initHelper(_numTables, _rangePow, _reservoirSize);

    std::cout << "LSH Reservoir Initialized in Node " << _myRank << std::endl;
}

void LSH::initVariables(unsigned int numHashPerFamily, unsigned int numHashFamilies,
                        unsigned int reservoirSize, unsigned int dimension,
                        unsigned int maxSamples) {
    _rangePow = numHashPerFamily;
    _numTables = numHashFamilies;
    _reservoirSize = reservoirSize;
    _dimension = dimension;
    _maxSamples = maxSamples;
    // _segmentSizeModulor = numHashFamilies * reservoirSize - 1;
    // _segmentSizeBitShiftDivisor = getLog2(_segmentSizeModulor);

    _numReservoirs = (unsigned int)pow(2, _rangePow); // Number of rows in each hashTable.
    // _numReservoirsHashed = (unsigned int)pow(2, _numSecHash); // Number of rows in each
    // hashTable.
    _maxReservoirRand = (unsigned int)ceil(maxSamples / 10); // TBD.
}

void LSH::initHelper(unsigned int numTablesIn, unsigned int numHashPerFamilyIn,
                     unsigned int reservoriSizeIn) {

    // Random initialization of hash functions
    // srand(time(NULL));

    // Fixed random seeds for hash functions
    srand(712376);
    _sechash_a = rand() * 2 + 1;
    _sechash_b = rand();

    _global_rand = new unsigned int[_maxReservoirRand];

    _global_rand[0] = 0;
    for (unsigned int i = 1; i < _maxReservoirRand; i++) {
        _global_rand[i] = rand() % i;
    }

    /* Hash tables. */
    _tableMemReservoirMax = _numTables * _numReservoirs;
    _tableMemMax = _tableMemReservoirMax * (1 + _reservoirSize);
    _tablePointerMax = _numTables * _numReservoirs;

    _tableMem = new unsigned int[_tableMemMax]();
    _tableMemAllocator = new unsigned int[_numTables]();
    _tablePointers = new unsigned int[_tablePointerMax];
    _tablePointersLock = new omp_lock_t[_tablePointerMax];
    for (unsigned long long i = 0; i < _tablePointerMax; i++) {
        _tablePointers[i] = TABLENULL;
        omp_init_lock(_tablePointersLock + i);
    }
    _tableCountersLock = new omp_lock_t[_tableMemReservoirMax];
    for (unsigned long long i = 0; i < _tableMemReservoirMax; i++) {
        omp_init_lock(_tableCountersLock + i);
    }
    /* Hashing counter. */
    _sequentialIDCounter_kernel = 0;
}

LSH::~LSH() {

    delete[] _tableMem;
    delete[] _tablePointers;
    delete[] _tableMemAllocator;
    for (size_t i = 0; i < _tablePointerMax; i++) {
        omp_destroy_lock(_tablePointersLock + i);
    }
    for (size_t i = 0; i < _tableMemReservoirMax; i++) {
        omp_destroy_lock(_tableCountersLock + i);
    }
    delete[] _tablePointersLock;
    delete[] _tableCountersLock;
    delete[] _global_rand;
}
