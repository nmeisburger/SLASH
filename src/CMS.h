#ifndef CMS_H_
#define CMS_H_

#define TIMER

#include <algorithm>
#include <cmath>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#define INT_MAX 0xffffff
#define TABLENULL -1

#define hashLocation(dataIndx, numHashes, hashIndx) ((int)(dataIndx * numHashes + hashIndx))
#define heavyHitterIndx(sketchIndx, sketchSize, bucketSize, hashIndx, hash)                        \
    ((int)(sketchIndx * sketchSize + bucketSize * 2 * hashIndx + hash * 2))
#define countIndx(sketchIndx, sketchSize, bucketSize, hashIndx, hash)                              \
    ((int)(sketchIndx * sketchSize + bucketSize * 2 * hashIndx + hash * 2 + 1))
#define sketchIndx(sketchIndx, sketchSize) ((int)(sketchIndx * sketchSize))

struct LHH {
    unsigned int heavyHitter;
    int count;
};

class CMS {
  private:
    unsigned int *_LHH;
    int _myRank, _worldSize;
    unsigned int _numHashes, _bucketSize, _numSketches;
    unsigned int _sketchSize;
    unsigned int *_hashingSeeds;

    void getCanidateHashes(unsigned int candidate, unsigned int *hashes);
    void getHashes(unsigned int *data, unsigned int dataSize, unsigned int *hashIndices);

    /* Adds a stream of data to the sketch.

    @param dataStreamIndx: The index of the data stream.
            The nth data stream will be added to the nth sketch.
    @param dataStream: The data to be added to the sketch.
    @param dataStreamLen: The length of the data to be added.
    */
    void addSketch(unsigned int dataStreamIndx, unsigned int *dataStream,
                   unsigned int dataStreamLen);

    /* Selects the top-k elements from the sketch.

    @param K: The number of top elements to return.
    @param threshold: The minimum count that a local heavy hitter must have
    to be added to the top K candidate set.
    @param sketchIndx: The index of the sketch to select the top-k from.
    */
    void topKSketch(unsigned int K, unsigned int threshold, unsigned int *topK,
                    unsigned int sketchIndx);

    /* Aggregates another sketch with the one stored in the object calling
    the function.
    





    @param newLHH: the new LHH sketch to combine with the existing.
    */
    void combineSketches(unsigned int *newLHH);

  public:
    /* Constructor

    @param L: The number of hashes per sketch
    @param B: The size of each bucket per hash function per sketch
    @param S: The number of sketches to initialize
    @param myRank: The rank of the node calling the constructor.
    @param worldSize: The total number of nodes.
    */
    CMS(unsigned int L, unsigned int B, unsigned int S, int myRank, int worldSize);

    /* Adds to each of the n sketches with n data streams

    @param dataStreams: The array holding all of the data streams to be
    added.
    @param segmentSize: The length of each datastream to be added.
    */
    void add(unsigned int *dataStreams, unsigned int segmentSize);

    /* Selects the top-k elements from every sketch

    @param: topK: The number of top elements to return from each sketch.
    @param outputs: The array to store the top-k selections in.
    @param threshold: The minimum count that a local heavy hitter must have
    to be added to the top K candidate set for each sketch.
    */
    void topK(unsigned int topK, unsigned int *outputs, unsigned int threshold);

    /* Aggregates sketches across all nodes by sending sketches to node 0
       where they are combined. Uses combineSketches helper function.
    */
    void aggregateSketches();

    void aggregateSketchesTree();

    /* For Debugging. Prints the heavy hitter and count for each element of
    the bucket for each hash function.

    @param sketchIndx: the index of the sketch to print the local heavy
    hitters and their counts from.
    */
    void showCMS(unsigned int sketchIndx);

    ~CMS();
};

#endif
