#include "LSH.h"

LSH::LSH(DOPH *hashFamIn, unsigned int numHashPerFamily, unsigned int numHashFamilies,
         unsigned int reservoirSize, unsigned int dimension, unsigned int maxSamples, int myRank,
         int worldSize) {

    _myRank = myRank;
    _worldSize = worldSize;

    _rangePow = numHashPerFamily;
    _numTables = numHashFamilies;
    _reservoirSize = reservoirSize;
    _dimension = dimension;
    _maxSamples = maxSamples;
    _numReservoirs = (unsigned int)pow(2, _rangePow);
    _maxReservoirRand = (unsigned int)ceil(maxSamples / 10);

    _hashFamily = hashFamIn;

    // Random initialization of hash functions
    // srand(time(NULL));

    // Fixed random seeds for hash functions
    srand(712376);

    _global_rand = new unsigned int[_maxReservoirRand];

    _global_rand[0] = 0;
    for (unsigned int i = 1; i < _maxReservoirRand; i++) {
        _global_rand[i] = rand() % i;
    }

    size_t tempNumTables = _numTables;
    size_t tempReservoirSize = _reservoirSize + 1;
    size_t tempNumReservoirs = _numReservoirs;

    size_t totalMem = tempNumReservoirs * tempNumTables * tempReservoirSize;

    _tableMem = new unsigned int[totalMem]();
    _reservoirs = new unsigned int *[_numTables * _numReservoirs];
    _reservoirsLock = new omp_lock_t[_numTables * _numReservoirs];
    for (size_t i = 0; i < _numTables * _numReservoirs; i++) {
        _reservoirs[i] = _tableMem + i * tempReservoirSize;
        omp_init_lock(_reservoirsLock + i);
    }
    /* Hashing counter. */
    _sequentialIDCounter_kernel = 0;

    std::cout << "LSH Reservoir Initialized in Node " << _myRank << std::endl;
}

LSH::~LSH() {

    delete[] _tableMem;
    delete[] _reservoirs;
    for (size_t i = 0; i < _numTables * _reservoirSize; i++) {
        omp_destroy_lock(_reservoirsLock + i);
    }
    delete[] _reservoirsLock;
    delete[] _global_rand;
}
