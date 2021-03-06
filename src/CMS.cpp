#include "CMS.h"

CMS::CMS(unsigned int L, unsigned int B, unsigned int numDataStreams, int myRank, int worldSize) {
    _numHashes = L;
    _bucketSize = B;
    _myRank = myRank;
    _worldSize = worldSize;
    _numSketches = numDataStreams;
    _sketchSize = _numHashes * _bucketSize * 2;
    _LHH = new unsigned int[_numSketches * _sketchSize]();
    _hashingSeeds = new unsigned int[_numHashes];

    if (_myRank == 0) {
        // Random hash functions
        // srand(time(NULL));

        // Fixed random seeds for hash functions
        srand(8524023);

        for (unsigned int h = 0; h < _numHashes; h++) {
            _hashingSeeds[h] = rand();
            if (_hashingSeeds[h] % 2 == 0) {
                _hashingSeeds[h]++;
            }
        }
    }

    MPI_Bcast(_hashingSeeds, _numHashes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::cout << "CMS Initialized in Node " << _myRank << std::endl;
}

void CMS::reset() {
    delete[] _LHH;
    _LHH = new unsigned int[_numSketches * _sketchSize]();
}

void CMS::getCanidateHashes(unsigned int candidate, unsigned int *hashes) {
    for (size_t hashIndx = 1; hashIndx < _numHashes; hashIndx++) {
        unsigned int h = _hashingSeeds[hashIndx];
        unsigned int k = candidate;
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        unsigned int curhash = (unsigned int)h % _bucketSize;
        hashes[hashIndx] = curhash;
    }
}

void CMS::getHashes(unsigned int *dataStream, unsigned int dataStreamLen,
                    unsigned int *hashIndices) {

#pragma omp parallel for default(none) shared(dataStream, hashIndices, dataStreamLen)
    for (size_t dataIndx = 0; dataIndx < dataStreamLen; dataIndx++) {
        for (size_t hashIndx = 0; hashIndx < _numHashes; hashIndx++) {
            unsigned int h = _hashingSeeds[hashIndx];
            unsigned int k = (unsigned int)dataStream[dataIndx];
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;
            h ^= k;
            h = (h << 13) | (h >> 19);
            h = h * 5 + 0xe6546b64;
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;
            unsigned int curhash = (unsigned int)h % _bucketSize;
            hashIndices[hashLocation(dataIndx, _numHashes, hashIndx)] = curhash;
        }
    }
}

void CMS::addSketch(unsigned int dataStreamIndx, unsigned int *dataStream,
                    unsigned int dataStreamLen) {

    // unsigned int *hashIndices = new unsigned int[_numHashes * dataStreamLen];
    unsigned int *hashIndices =
        (unsigned int *)malloc(sizeof(unsigned int) * _numHashes * dataStreamLen);
    getHashes(dataStream, dataStreamLen, hashIndices);

    for (size_t dataIndx = 0; dataIndx < dataStreamLen; dataIndx++) {
        for (size_t hashIndx = 0; hashIndx < _numHashes; hashIndx++) {
            if (dataStream[dataIndx] == 0) {
                continue;
            }
            unsigned int currentHash = hashIndices[hashLocation(dataIndx, _numHashes, hashIndx)];
            unsigned int *LHH_ptr = _LHH + heavyHitterIndx(dataStreamIndx, _sketchSize, _bucketSize,
                                                           hashIndx, currentHash);
            unsigned int *LHH_Count_ptr =
                _LHH + countIndx(dataStreamIndx, _sketchSize, _bucketSize, hashIndx, currentHash);
            if (*LHH_Count_ptr != 0) {
                if (dataStream[dataIndx] == *LHH_ptr) {
                    *LHH_Count_ptr = *LHH_Count_ptr + 1;
                } else {
                    *LHH_Count_ptr = *LHH_Count_ptr - 1;
                }
            }
            if (*LHH_Count_ptr == 0) {
                *LHH_ptr = (int)dataStream[dataIndx];
                *LHH_Count_ptr = 1;
            }
        }
    }
    // delete[] hashIndices;
    free(hashIndices);
}

void CMS::add(unsigned int *dataStreams, unsigned int segmentSize) {

#pragma omp parallel for default(none) shared(dataStreams, segmentSize)
    for (unsigned int streamIndx = 0; streamIndx < _numSketches; streamIndx++) {
        addSketch(streamIndx, dataStreams + streamIndx * segmentSize, segmentSize);
    }
}

void CMS::topKSketch(unsigned int K, unsigned int threshold, unsigned int *topK,
                     unsigned int sketchIndx) {

    LHH *candidates = new LHH[_bucketSize];
    int count = 0;
    for (int b = 0; b < _bucketSize; b++) {
        unsigned int currentHeavyHitter =
            _LHH[heavyHitterIndx(sketchIndx, _sketchSize, _bucketSize, 0, b)];
        unsigned int currentCount = _LHH[countIndx(sketchIndx, _sketchSize, _bucketSize, 0, b)];
        if (currentCount >= threshold) {
            candidates[count].heavyHitter = currentHeavyHitter;
            candidates[count].count = currentCount;
            count++;
        } else {
            unsigned int *hashes = new unsigned int[_numHashes];
            getCanidateHashes(currentHeavyHitter, hashes);
            for (unsigned int hashIndx = 1; hashIndx < _numHashes; hashIndx++) {
                currentCount = _LHH[countIndx(sketchIndx, _sketchSize, _bucketSize, hashIndx,
                                              hashes[hashIndx])];
                if (currentCount > threshold) {
                    candidates[count].heavyHitter = currentHeavyHitter;
                    candidates[count].count = currentCount;
                    count++;
                    break;
                }
            }
            delete[] hashes;
        }
    }
    for (; count < _bucketSize; count++) {
        candidates[count].heavyHitter = -1;
        candidates[count].count = -1;
    }
    std::sort(candidates, candidates + _bucketSize,
              [&candidates](LHH a, LHH b) { return a.count > b.count; });

    for (size_t i = 0; i < K; i++) {
        if (candidates[i].count > -1) {
            topK[i] = candidates[i].heavyHitter;
        }
    }
    delete[] candidates;
}

void CMS::topK(unsigned int topK, unsigned int *outputs, unsigned int threshold) {

#pragma omp parallel for default(none) shared(topK, outputs, threshold)
    for (unsigned int sketchIndx = 0; sketchIndx < _numSketches; sketchIndx++) {
        topKSketch(topK, threshold, outputs + sketchIndx * topK, sketchIndx);
    }
}

void CMS::combineSketches(unsigned int *newLHH) {

// #pragma omp parallel for default(none) shared(newLHH, _LHH, _numHashes, _bucketSize,
// _numSketches)
#pragma omp parallel for default(none) shared(newLHH)
    for (size_t n = 0; n < _numHashes * _bucketSize * _numSketches; n++) {
        // if (newLHH[n * 2] == _LHH[n * 2]) {
        //     _LHH[n * 2 + 1] += newLHH[n * 2 + 1];
        // } else {
        //     _LHH[n * 2 + 1] -= newLHH[n * 2 + 1];
        // }
        // if (_LHH[n * 2 + 1] <= 0) {
        //     _LHH[n * 2] = newLHH[n * 2];
        //     _LHH[n * 2 + 1] = -_LHH[n * 2 + 1];
        // }
        if (newLHH[n * 2] == _LHH[n * 2]) {
            _LHH[n * 2 + 1] += newLHH[n * 2 + 1];
        } else {
            if (_LHH[n * 2 + 1] > newLHH[n * 2 + 1]) {
                _LHH[n * 2 + 1] -= newLHH[n * 2 + 1];
            } else {
                _LHH[n * 2] = newLHH[n * 2];
                _LHH[n * 2 + 1] = newLHH[n * 2 + 1] - _LHH[n * 2 + 1];
            }
        }
    }
}

void CMS::aggregateSketches() {
    unsigned long long bufferSize = _sketchSize * _numSketches;
    unsigned int *sketchBuffer;
    if (_myRank == 0) {
        sketchBuffer = new unsigned int[bufferSize * (unsigned long long)_worldSize];
    }
    MPI_Gather(_LHH, bufferSize, MPI_UNSIGNED, sketchBuffer, bufferSize, MPI_UNSIGNED, 0,
               MPI_COMM_WORLD);
    if (_myRank == 0) {
        for (int n = 1; n < _worldSize; n++) {
            combineSketches(sketchBuffer + (n * bufferSize));
        }
        delete[] sketchBuffer;
    }
}

void CMS::aggregateSketchesTree() {
    unsigned long long bufferSize = _sketchSize * _numSketches;
    unsigned int numIterations = std::ceil(std::log(_worldSize) / std::log(2));
    unsigned int *recvBuffer = new unsigned int[bufferSize];
    MPI_Status status;
    for (unsigned int iter = 0; iter < numIterations; iter++) {
        if (_myRank % ((int)std::pow(2, iter + 1)) == 0 &&
            (_myRank + std::pow(2, iter)) < _worldSize) {
            int source = _myRank + std::pow(2, iter);
            MPI_Recv(recvBuffer, bufferSize, MPI_UNSIGNED, source, iter, MPI_COMM_WORLD, &status);
            combineSketches(recvBuffer);
            // printf("Iteration %d: Node %d: Recv from %d\n", iter, _myRank, source);
        } else if (_myRank % ((int)std::pow(2, iter + 1)) == ((int)std::pow(2, iter))) {
            int destination = _myRank - ((int)std::pow(2, iter));
            MPI_Send(_LHH, bufferSize, MPI_UNSIGNED, destination, iter, MPI_COMM_WORLD);
            // printf("Iteration %d: Node %d: Send from %d\n", iter, _myRank, destination);
        }
    }
    delete[] recvBuffer;
}

void CMS::showCMS(unsigned int sketchIndx) {
    for (size_t l = 0; l < _numHashes; l++) {
        printf("Bucket %zu:\n\t[LHH]: ", l);
        for (int b = 0; b < _bucketSize; b++) {
            printf("\t%d", _LHH[heavyHitterIndx(sketchIndx, _sketchSize, _bucketSize, l, b)]);
        }
        printf("\n\t[Cnt]: ");
        for (int b = 0; b < _bucketSize; b++) {
            printf("\t%d", _LHH[countIndx(sketchIndx, _sketchSize, _bucketSize, l, b)]);
        }
        printf("\n");
    }
}

CMS::~CMS() {

    delete[] _LHH;
    delete[] _hashingSeeds;
}
