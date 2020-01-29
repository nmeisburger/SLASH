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

    // add(num_vectors, data_markers, data_indices, node_offsets[_my_rank] +
    // _total_vectors_added);

    _total_vectors_added += num_vectors;

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

    unsigned int *query_hash_buffer = new unsigned int[num_vectors * _L];
    unsigned int *all_query_hashes = new unsigned int[num_vectors * _L];

    MPI_Allgatherv(my_query_hashes, node_vector_counts[_my_rank] * _L, MPI_UNSIGNED,
                   all_query_hashes, hash_counts, hash_offsets, MPI_UNSIGNED, MPI_COMM_WORLD);

    unsigned int len;

    unsigned int *old;
    unsigned int *fin;

    // #pragma omp parallel for default(none)                                                             \
//     shared(query_hash_buffer, all_query_hashes, hash_offsets, num_vectors,                         \
//            node_vector_counts) private(len, old, fin)
    //     for (unsigned int partition = 0; partition < _world_size;
    //     partition++) {
    //         len = node_vector_counts[partition];
    //         for (unsigned int tb = 0; tb < _L; tb++) {
    //             old = query_hash_buffer + hash_offsets[partition] + tb *
    //             len; fin = all_query_hashes + tb * num_vectors +
    //             (hash_offsets[partition] / _L); for (int l = 0; l < len;
    //             l++) {
    //                 fin[l] = old[l];
    //             }
    //         }
    //     }

    long segment_size = _L * _reservoir_size;
    unsigned int *extracted_reservoirs = new unsigned int[segment_size * (long)num_vectors]();
    extract(num_vectors, all_query_hashes, extracted_reservoirs);

    // for (int v = 0; v < num_vectors; v++) {
    //     printf("\nQuery %d:\n", v);
    //     for (int t = 0; t < _L; t++) {
    //         printf("Table %d: ", t);
    //         for (int i = 0; i < _reservoir_size; i++) {
    //             if (extracted_reservoirs[EXTRACTED_INDEX(_reservoir_size,
    //             _L, v, t, i)] != INT_MAX)
    //                 printf(" %u ",
    //                        extracted_reservoirs[EXTRACTED_INDEX(_reservoir_size,
    //                        _L, v, t, i)]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < segment_size * num_vectors; i++) {
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

    delete[] query_hash_buffer;
    delete[] all_query_hashes;

    delete[] extracted_reservoirs;
}