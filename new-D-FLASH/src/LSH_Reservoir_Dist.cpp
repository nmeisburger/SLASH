#include "LSH_Reservoir.h"

void LSH_Reservoir::add_dist(std::string filename, unsigned int read_offset,
                             unsigned int num_vectors, unsigned int dimension) {

    unsigned int node_offsets[_world_size];
    unsigned int node_vector_counts[_world_size];

    unsigned int data_partition_size =
        std::floor((float)num_vectors / (float)_world_size); // fix to use int division
    unsigned int data_partition_remainder = num_vectors % _world_size;

    for (int i = 0; i < _world_size; i++) {
        node_vector_counts[i] = data_partition_size;
        if (i < data_partition_remainder) {
            node_vector_counts[i]++;
        }
    }

    node_offsets[0] = read_offset;
    for (int i = 1; i < _world_size; i++) {
        node_offsets[i] = std::min(node_offsets[i - 1] + node_vector_counts[i - 1],
                                   num_vectors + read_offset - 1);
    }

    printf("Node %d - Add Counts %d @ offset %d\n", _my_rank, node_vector_counts[_my_rank],
           node_offsets[_my_rank]);

    unsigned int *data_markers = new unsigned int[node_vector_counts[_my_rank] + 1];
    unsigned int *data_indices = new unsigned int[node_vector_counts[_my_rank] * dimension];
    float *data_values = new float[node_vector_counts[_my_rank] * dimension];

    read_sparse(filename, node_offsets[_my_rank], node_vector_counts[_my_rank], data_indices,
                data_values, data_markers, node_vector_counts[_my_rank] * dimension);

    add(node_vector_counts[_my_rank], data_markers, data_indices,
        node_offsets[_my_rank] + _total_vectors_added);

    _total_vectors_added += num_vectors;

    printf("\n <<<< Data Vectors Added Node %d >>>>>\n", _my_rank);

    delete[] data_markers;
    delete[] data_indices;
    delete[] data_values;
}

void LSH_Reservoir::query_dist(std::string filename, unsigned int read_offset,
                               unsigned int num_vectors, unsigned int dimension, int top_k,
                               unsigned int *outputs) {
    unsigned int node_offsets[_world_size];
    unsigned int node_vector_counts[_world_size];

    unsigned int query_partition_size =
        std::floor((float)num_vectors / (float)_world_size); // fix to use int division
    unsigned int query_partition_remainder = num_vectors % _world_size;

    for (int i = 0; i < _world_size; i++) {
        node_vector_counts[i] = query_partition_size;
        if (i < query_partition_remainder) {
            node_vector_counts[i]++;
        }
    }

    unsigned int *query_markers = new unsigned int[node_vector_counts[_my_rank] + 1];
    unsigned int *query_indices = new unsigned int[node_vector_counts[_my_rank] * dimension];
    float *query_values = new float[node_vector_counts[_my_rank] * dimension];

    node_offsets[0] = read_offset;
    for (int i = 1; i < _world_size; i++) {
        node_offsets[i] = std::min(node_offsets[i - 1] + node_vector_counts[i - 1],
                                   num_vectors + read_offset - 1);
    }

    int *hash_offsets = new int[_world_size];
    int *hash_counts = new int[_world_size];

    for (int i = 0; i < _world_size; i++) {
        hash_counts[i] = node_vector_counts[0] * _L;
        hash_offsets[i] = (node_offsets[i] - read_offset) * _L;
    }

    read_sparse(filename, node_offsets[_my_rank], node_vector_counts[_my_rank], query_indices,
                query_values, query_markers, node_vector_counts[_my_rank] * dimension);

    unsigned int *my_query_hashes = new unsigned int[node_vector_counts[_my_rank] * _L];
    _hash_family->hash(node_vector_counts[_my_rank], query_indices, query_markers, my_query_hashes);

    // MPI_Barrier(MPI_COMM_WORLD);
    // for (int i = 0; i < _world_size; i++) {
    //     if (_my_rank == i) {
    //         for (int v = 0; v < node_vector_counts[_my_rank]; v++) {
    //             printf("Node %d Vector: %d: ", i, v);
    //             for (int t = 0; t < _L; t++) {
    //                 printf("%d\t", my_query_hashes[HASH_OUTPUT_INDEX(_L, v, t)]);
    //             }
    //             printf("\n");
    //         }
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    unsigned int *all_query_hashes = new unsigned int[num_vectors * _L];

    MPI_Allgatherv(my_query_hashes, node_vector_counts[_my_rank] * _L, MPI_UNSIGNED,
                   all_query_hashes, hash_counts, hash_offsets, MPI_UNSIGNED, MPI_COMM_WORLD);

    printf("\n <<<< Query Hashes Computed Node %d >>>>>\n", _my_rank);
    // if (_my_rank == 0) {
    //     for (int v = 0; v < num_vectors; v++) {
    //         printf("Vector: %d: ", v);
    //         for (int t = 0; t < _L; t++) {
    //             printf("%d\t", all_query_hashes[HASH_OUTPUT_INDEX(_L, v, t)]);
    //         }
    //         printf("\n");
    //     }
    // }

    long segment_size = _L * _reservoir_size;
    unsigned int *extracted_reservoirs = new unsigned int[segment_size * (long)num_vectors];
    for (int i = 0; i < segment_size * num_vectors; i++) {
        extracted_reservoirs[i] = INT_MAX;
    }
    extract(num_vectors, all_query_hashes, extracted_reservoirs);

    printf("\n <<<< Query Extracted Node %d >>>>>\n", _my_rank);

    for (int n = 0; n < _world_size; n++) {
        if (_my_rank == n) {
            for (int i = 0; i < node_vector_counts; i++) {
                printf("NODE %d VECTOR %d", n, i);
                for (int j = 0; j < segment_size; j++) {
                    unsigned int x = extracted_reservoirs[i * segment_size + j];
                    if (x != INT_MAX) {
                        printf(" %d ", x);
                    }
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    _top_k_sketch->add(extracted_reservoirs, segment_size);

    _top_k_sketch->aggregateSketchesTree();

    if (_my_rank == 0) {
        _top_k_sketch->topK(top_k, outputs, 0);
    }

    delete[] query_markers;
    delete[] query_indices;
    delete[] query_values;

    delete[] hash_counts;
    delete[] hash_offsets;

    delete[] all_query_hashes;

    delete[] extracted_reservoirs;
}