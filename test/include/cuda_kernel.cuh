#pragma once

#include "batch.hpp"
#include "cudautility.hpp"
#include "cuda_sw.cuh"
#include "cuda_topologicalsort.cuh"
#include "cuda_add_alignment.cuh"
#include "cuda_generate_consensus.cuh"

// considering 65536 register file size, __launch_bounds__(1024) translates to max 64 register usage
#define GW_POA_KERNELS_MAX_THREADS_PER_BLOCK_64_REGISTERS 1024
#define GW_POA_KERNELS_MAX_THREADS_PER_BLOCK_72_REGISTERS 896


/**
 * @brief The main kernel that runs the partial order alignment
 *        algorithm.
 *
 * @param[out] consensus_d                  Device buffer for generated consensus
 * @param[in] sequences_d                   Device buffer with sequences for all windows
 * @param[in] base_weights_d                Device buffer with base weights for all windows
 * @param[in] sequence_lengths_d            Device buffer sequence lengths
 * @param[in] window_details_d              Device buffer with structs encapsulating sequence details per window
 * @param[in] total_windows                 Total number of windows to process
 * @param[in] scores_d                      Device scratch space that scores alignment matrix score
 * @param[in] alignment_graph_d             Device scratch space for traceback alignment of graph
 * @param[in] alignment_read_d              Device scratch space for traceback alignment of sequence
 * @param[in] nodes_d                       Device scratch space for storing unique nodes in graph
 * @param[in] incoming_edges_d              Device scratch space for storing incoming edges per node
 * @param[in] incoming_edges_count_d        Device scratch space for storing number of incoming edges per node
 * @param[in] outgoing_edges_d              Device scratch space for storing outgoing edges per node
 * @param[in] outgoing_edges_count_d        Device scratch space for storing number of outgoing edges per node
 * @param[in] incoming_edge_w_d             Device scratch space for storing weight of incoming edges
 * @param[in] sorted_poa_d                  Device scratch space for storing sorted graph
 * @param[in] node_id_to_pos_d              Device scratch space for mapping node ID to position in graph
 * @graph[in] node_alignments_d             Device scratch space for storing alignment nodes per node in graph
 * @param[in] node_alignment_count_d        Device scratch space for storing number of aligned nodes
 * @param[in] sorted_poa_local_edge_count_d Device scratch space for maintaining edge counts during topological sort
 * @param[in] node_marks_d_                 Device scratch space for storing node marks when running spoa accurate top sort
 * @param[in] check_aligned_nodes_d_        Device scratch space for storing check for aligned nodes
 * @param[in] nodes_to_visit_d_             Device scratch space for storing stack of nodes to be visited in topsort
 * @param[in] node_coverage_counts_d_       Device scratch space for storing coverage of each node in graph.
 * @param[in] gap_score                     Score for inserting gap into alignment
 * @param[in] mismatch_score                Score for finding a mismatch in alignment
 * @param[in] match_score                   Score for finding a match in alignment
 */
template <typename ScoreT, typename SizeT, typename TraceT, bool MSA = false, BandMode BM = full_band, bool Traceback = false>
__launch_bounds__(Traceback ? GW_POA_KERNELS_MAX_THREADS_PER_BLOCK_72_REGISTERS : GW_POA_KERNELS_MAX_THREADS_PER_BLOCK_64_REGISTERS)
    __global__ void generatePOAKernel(uint8_t* consensus_d,
                                      uint8_t* sequences_d,
                                      int8_t* base_weights_d,
                                      SizeT* sequence_lengths_d,
                                      WindowDetails* window_details_d,
                                      int32_t total_windows,
                                      ScoreT* scores_d,
                                      SizeT* alignment_graph_d,
                                      SizeT* alignment_read_d,
                                      uint8_t* nodes_d,
                                      SizeT* incoming_edges_d,
                                      uint16_t* incoming_edge_count_d,
                                      SizeT* outgoing_edges_d,
                                      uint16_t* outgoing_edge_count_d,
                                      uint16_t* incoming_edge_w_d,
                                      SizeT* sorted_poa_d,
                                      SizeT* node_id_to_pos_d,
                                      SizeT* node_alignments_d,
                                      uint16_t* node_alignment_count_d,
                                      uint16_t* sorted_poa_local_edge_count_d,
                                      uint8_t* node_marks_d_,
                                      bool* check_aligned_nodes_d_,
                                      SizeT* nodes_to_visit_d_,
                                      uint16_t* node_coverage_counts_d_,
                                      int32_t gap_score,
                                      int32_t mismatch_score,
                                      int32_t match_score,
                                      uint32_t max_sequences_per_poa,
                                      SizeT* sequence_begin_nodes_ids_d,
                                      uint16_t* outgoing_edges_coverage_d,
                                      uint16_t* outgoing_edges_coverage_count_d,
                                      int32_t max_nodes_per_graph,
                                      int32_t scores_matrix_width,
                                      int32_t max_limit_consensus_size,
                                      int32_t TPB               = 64,
                                      int32_t static_band_width = 256,
                                      int32_t max_pred_distance = 0,
                                      TraceT* traceback_d       = nullptr)
{
    // shared error indicator within a warp
    bool warp_error = false;

    uint32_t lane_idx   = threadIdx.x % WARP_SIZE;
    uint32_t window_idx = blockIdx.x * TPB / WARP_SIZE + threadIdx.x / WARP_SIZE;

    if (window_idx >= total_windows)
        return;

    // Find the buffer offsets for each thread within the global memory buffers.
    uint8_t* nodes                        = &nodes_d[max_nodes_per_graph * window_idx];
    SizeT* incoming_edges                 = &incoming_edges_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* incoming_edge_count         = &incoming_edge_count_d[window_idx * max_nodes_per_graph];
    SizeT* outgoing_edges                 = &outgoing_edges_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_count         = &outgoing_edge_count_d[window_idx * max_nodes_per_graph];
    uint16_t* incoming_edge_weights       = &incoming_edge_w_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    SizeT* sorted_poa                     = &sorted_poa_d[window_idx * max_nodes_per_graph];
    SizeT* node_id_to_pos                 = &node_id_to_pos_d[window_idx * max_nodes_per_graph];
    SizeT* node_alignments                = &node_alignments_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_ALIGNMENTS];
    uint16_t* node_alignment_count        = &node_alignment_count_d[window_idx * max_nodes_per_graph];
    uint16_t* sorted_poa_local_edge_count = &sorted_poa_local_edge_count_d[window_idx * max_nodes_per_graph];

    ScoreT* scores;
    float banded_buffer_size; // using float instead of int64_t to minimize register

    TraceT* traceback    = traceback_d;                               // only used in traceback
    int32_t scores_width = window_details_d[window_idx].scores_width; // only used in non-traceback

    if (Traceback)
    {
        // buffer size for scores, in traceback we only need to store part of the scores matrix
        banded_buffer_size = static_cast<float>(max_pred_distance) * static_cast<float>(scores_matrix_width);
        int64_t offset     = static_cast<int64_t>(banded_buffer_size) * static_cast<int64_t>(window_idx);
        scores             = &scores_d[offset];
        // buffer size for traceback
        banded_buffer_size = static_cast<float>(max_nodes_per_graph) * static_cast<float>(scores_matrix_width);
        offset             = static_cast<int64_t>(banded_buffer_size) * static_cast<int64_t>(window_idx);
        traceback          = &traceback_d[offset];
    }
    else
    {
        if (BM == BandMode::adaptive_band || BM == BandMode::static_band)
        {
            banded_buffer_size    = static_cast<float>(max_nodes_per_graph) * static_cast<float>(scores_matrix_width);
            int64_t scores_offset = static_cast<int64_t>(banded_buffer_size) * static_cast<int64_t>(window_idx);
            scores                = &scores_d[scores_offset];
        }
        else if (BM == BandMode::full_band)
        {
            int64_t offset = static_cast<int64_t>(window_details_d[window_idx].scores_offset) * static_cast<int64_t>(max_nodes_per_graph);
            scores         = &scores_d[offset];
        }
    }

    SizeT* alignment_graph         = &alignment_graph_d[max_nodes_per_graph * window_idx];
    SizeT* alignment_read          = &alignment_read_d[max_nodes_per_graph * window_idx];
    uint16_t* node_coverage_counts = &node_coverage_counts_d_[max_nodes_per_graph * window_idx];

// #ifdef SPOA_ACCURATE
//     uint8_t* node_marks       = &node_marks_d_[max_nodes_per_graph * window_idx];
//     bool* check_aligned_nodes = &check_aligned_nodes_d_[max_nodes_per_graph * window_idx];
//     SizeT* nodes_to_visit     = &nodes_to_visit_d_[max_nodes_per_graph * window_idx];
// #endif

    SizeT* sequence_lengths = &sequence_lengths_d[window_details_d[window_idx].seq_len_buffer_offset];

    uint16_t num_sequences = window_details_d[window_idx].num_seqs;
    uint8_t* sequence      = &sequences_d[window_details_d[window_idx].seq_starts];
    int8_t* base_weights   = &base_weights_d[window_details_d[window_idx].seq_starts];

    uint8_t* consensus = &consensus_d[window_idx * max_limit_consensus_size];

    SizeT* sequence_begin_nodes_ids         = nullptr;
    uint16_t* outgoing_edges_coverage       = nullptr;
    uint16_t* outgoing_edges_coverage_count = nullptr;

    if (MSA)
    {
        sequence_begin_nodes_ids      = &sequence_begin_nodes_ids_d[window_idx * max_sequences_per_poa];
        outgoing_edges_coverage       = &outgoing_edges_coverage_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa];
        outgoing_edges_coverage_count = &outgoing_edges_coverage_count_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    }

    if (lane_idx == 0)
    {
        // Create backbone for window based on first sequence in window.
        nodes[0]                                     = sequence[0];
        sorted_poa[0]                                = 0;
        incoming_edge_count[0]                       = 0;
        node_alignment_count[0]                      = 0;
        node_id_to_pos[0]                            = 0;
        outgoing_edge_count[sequence_lengths[0] - 1] = 0;
        incoming_edge_weights[0]                     = base_weights[0];
        node_coverage_counts[0]                      = 1;
        if (MSA)
        {
            sequence_begin_nodes_ids[0] = 0;
        }

        //Build the rest of the graphs
        for (SizeT nucleotide_idx = 1; nucleotide_idx < sequence_lengths[0]; nucleotide_idx++)
        {
            nodes[nucleotide_idx]                                          = sequence[nucleotide_idx];
            sorted_poa[nucleotide_idx]                                     = nucleotide_idx;
            outgoing_edges[(nucleotide_idx - 1) * CUDAPOA_MAX_NODE_EDGES]  = nucleotide_idx;
            outgoing_edge_count[nucleotide_idx - 1]                        = 1;
            incoming_edges[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES]        = nucleotide_idx - 1;
            incoming_edge_weights[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES] = base_weights[nucleotide_idx - 1] + base_weights[nucleotide_idx];
            incoming_edge_count[nucleotide_idx]                            = 1;
            node_alignment_count[nucleotide_idx]                           = 0;
            node_id_to_pos[nucleotide_idx]                                 = nucleotide_idx;
            node_coverage_counts[nucleotide_idx]                           = 1;
            if (MSA)
            {
                outgoing_edges_coverage[(nucleotide_idx - 1) * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa] = 0;
                outgoing_edges_coverage_count[(nucleotide_idx - 1) * CUDAPOA_MAX_NODE_EDGES]                   = 1;
            }
        }

        // Clear error code for window.
        consensus[0] = CUDAPOA_KERNEL_NOERROR_ENCOUNTERED;
    }

    __syncwarp();

    // Align each subsequent read, add alignment to graph, run topological sort.
    for (int32_t s = 1; s < num_sequences; s++)
    {
        int32_t seq_len = sequence_lengths[s];
        // Note: the following 2 lines correspond to num_nucleotides_copied_ value on the host side
        // therefore it is important to reflect any changes in the following 2 lines in the corresponding host code as well
        sequence += align<int32_t, SIZE_OF_SeqT4>(sequence_lengths[s - 1]);     // increment the pointer so it is pointing to correct sequence data
        base_weights += align<int32_t, SIZE_OF_SeqT4>(sequence_lengths[s - 1]); // increment the pointer so it is pointing to correct sequence data

        if (lane_idx == 0)
        {
            if (sequence_lengths[0] >= max_nodes_per_graph)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = static_cast<uint8_t>(StatusType::node_count_exceeded_maximum_graph_size);
                warp_error   = true;
            }
        }

        warp_error = __shfl_sync(FULL_MASK, warp_error, 0);
        if (warp_error)
        {
            return;
        }

        // Run Needleman-Wunsch alignment between graph and new sequence.
        SizeT alignment_length;

        alignment_length = needlemanWunsch<uint8_t, ScoreT, SizeT>(nodes,
                                                                           sorted_poa,
                                                                           node_id_to_pos,
                                                                           sequence_lengths[0],
                                                                           incoming_edge_count,
                                                                           incoming_edges,
                                                                           outgoing_edge_count,
                                                                           sequence,
                                                                           seq_len,
                                                                           scores,
                                                                           scores_width,
                                                                           alignment_graph,
                                                                           alignment_read,
                                                                           gap_score,
                                                                           mismatch_score,
                                                                           match_score);
        __syncwarp();

        if (alignment_length == CUDAPOA_KERNEL_NW_BACKTRACKING_LOOP_FAILED)
        {
            if (lane_idx == 0)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = static_cast<uint8_t>(StatusType::loop_count_exceeded_upper_bound);
            }
            return;
        }
        else if (alignment_length == CUDAPOA_KERNEL_NW_ADAPTIVE_STORAGE_FAILED)
        {
            if (lane_idx == 0)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = static_cast<uint8_t>(StatusType::exceeded_adaptive_banded_matrix_size);
            }
            return;
        }
        if (Traceback)
        {
            if (alignment_length == CUDAPOA_KERNEL_NW_TRACEBACK_BUFFER_FAILED)
            {
                if (lane_idx == 0)
                {
                    consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                    consensus[1] = static_cast<uint8_t>(StatusType::exceeded_maximum_predecessor_distance);
                }
                return;
            }
        }

        if (lane_idx == 0)
        {

            // Add alignment to graph.
            SizeT new_node_count;
            uint8_t error_code = addAlignmentToGraph<SizeT, MSA>(new_node_count,
                                                                 nodes, sequence_lengths[0],
                                                                 node_alignments, node_alignment_count,
                                                                 incoming_edges, incoming_edge_count,
                                                                 outgoing_edges, outgoing_edge_count,
                                                                 incoming_edge_weights,
                                                                 alignment_length,
                                                                 sorted_poa, alignment_graph,
                                                                 sequence, alignment_read,
                                                                 node_coverage_counts,
                                                                 base_weights,
                                                                 (sequence_begin_nodes_ids + s),
                                                                 outgoing_edges_coverage,
                                                                 outgoing_edges_coverage_count,
                                                                 s,
                                                                 max_sequences_per_poa,
                                                                 max_nodes_per_graph);

            if (error_code != 0)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = error_code;
                warp_error   = true;
            }
            else
            {
                sequence_lengths[0] = new_node_count;
                // Run a topsort on the graph.
#ifdef SPOA_ACCURATE
                // Exactly matches racon CPU results
                raconTopologicalSortDeviceUtil(sorted_poa,
                                               node_id_to_pos,
                                               new_node_count,
                                               incoming_edge_count,
                                               incoming_edges,
                                               node_alignment_count,
                                               node_alignments,
                                               node_marks,
                                               check_aligned_nodes,
                                               nodes_to_visit,
                                               (uint16_t)max_nodes_per_graph);
#else
                // Faster top sort
                topologicalSortDeviceUtil(sorted_poa,
                                          node_id_to_pos,
                                          new_node_count,
                                          incoming_edge_count,
                                          outgoing_edges,
                                          outgoing_edge_count,
                                          sorted_poa_local_edge_count);
#endif
            }
        }

        __syncwarp();

        warp_error = __shfl_sync(FULL_MASK, warp_error, 0);
        if (warp_error)
        {
            return;
        }
    }
}

template <typename ScoreT, typename SizeT, typename TraceT>
void generatePOA(OutputDetails* output_details_d,
                 InputDetails<SizeT>* input_details_d,
                 int32_t total_windows,
                 cudaStream_t stream,
                 AlignmentDetails<ScoreT, SizeT, TraceT>* alignment_details_d,
                 GraphDetails<SizeT>* graph_details_d,
                 int32_t gap_score,
                 int32_t mismatch_score,
                 int32_t match_score,
                 uint32_t max_sequences_per_poa,
                 int8_t output_mask,
                 const BatchConfig& batch_size)
{
    // unpack output details
    uint8_t* consensus_d                  = output_details_d->consensus;
    uint16_t* coverage_d                  = output_details_d->coverage;
    uint8_t* multiple_sequence_alignments = output_details_d->multiple_sequence_alignments;

    // unpack input details
    uint8_t* sequences_d            = input_details_d->sequences;
    int8_t* base_weights_d          = input_details_d->base_weights;
    SizeT* sequence_lengths_d       = input_details_d->sequence_lengths;
    WindowDetails* window_details_d = input_details_d->window_details;
    SizeT* sequence_begin_nodes_ids = input_details_d->sequence_begin_nodes_ids;

    // unpack alignment details
    ScoreT* scores         = alignment_details_d->scores;
    TraceT* traceback      = alignment_details_d->traceback;
    SizeT* alignment_graph = alignment_details_d->alignment_graph;
    SizeT* alignment_read  = alignment_details_d->alignment_read;

    // unpack graph details
    uint8_t* nodes                          = graph_details_d->nodes;
    SizeT* node_alignments                  = graph_details_d->node_alignments;
    uint16_t* node_alignment_count          = graph_details_d->node_alignment_count;
    SizeT* incoming_edges                   = graph_details_d->incoming_edges;
    uint16_t* incoming_edge_count           = graph_details_d->incoming_edge_count;
    SizeT* outgoing_edges                   = graph_details_d->outgoing_edges;
    uint16_t* outgoing_edge_count           = graph_details_d->outgoing_edge_count;
    uint16_t* incoming_edge_w               = graph_details_d->incoming_edge_weights;
    SizeT* sorted_poa                       = graph_details_d->sorted_poa;
    SizeT* node_id_to_pos                   = graph_details_d->sorted_poa_node_map;
    uint16_t* sorted_poa_local_edge_count   = graph_details_d->sorted_poa_local_edge_count;
    int32_t* consensus_scores               = graph_details_d->consensus_scores;
    SizeT* consensus_predecessors           = graph_details_d->consensus_predecessors;
    uint8_t* node_marks                     = graph_details_d->node_marks;
    bool* check_aligned_nodes               = graph_details_d->check_aligned_nodes;
    SizeT* nodes_to_visit                   = graph_details_d->nodes_to_visit;
    uint16_t* node_coverage_counts          = graph_details_d->node_coverage_counts;
    uint16_t* outgoing_edges_coverage       = graph_details_d->outgoing_edges_coverage;
    uint16_t* outgoing_edges_coverage_count = graph_details_d->outgoing_edges_coverage_count;
    SizeT* node_id_to_msa_pos               = graph_details_d->node_id_to_msa_pos;

    int32_t nwindows_per_block   = CUDAPOA_THREADS_PER_BLOCK / WARP_SIZE;
    int32_t nblocks              = (batch_size.band_mode != BandMode::full_band) ? total_windows : (total_windows + nwindows_per_block - 1) / nwindows_per_block;
    int32_t TPB                  = (batch_size.band_mode != BandMode::full_band) ? CUDAPOA_BANDED_THREADS_PER_BLOCK : CUDAPOA_THREADS_PER_BLOCK;
    int32_t max_nodes_per_graph  = batch_size.max_nodes_per_graph;
    int32_t matrix_seq_dimension = batch_size.matrix_sequence_dimension;
    bool msa                     = output_mask & OutputType::msa;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int32_t consensus_num_blocks = (total_windows / CUDAPOA_MAX_CONSENSUS_PER_BLOCK) + 1;

    generatePOAKernel<ScoreT, SizeT, TraceT, false, BandMode::full_band>
                <<<nblocks, TPB, 0, stream>>>(consensus_d,
                                              sequences_d,
                                              base_weights_d,
                                              sequence_lengths_d,
                                              window_details_d,
                                              total_windows,
                                              scores,
                                              alignment_graph,
                                              alignment_read,
                                              nodes,
                                              incoming_edges,
                                              incoming_edge_count,
                                              outgoing_edges,
                                              outgoing_edge_count,
                                              incoming_edge_w,
                                              sorted_poa,
                                              node_id_to_pos,
                                              node_alignments,
                                              node_alignment_count,
                                              sorted_poa_local_edge_count,
                                              node_marks,
                                              check_aligned_nodes,
                                              nodes_to_visit,
                                              node_coverage_counts,
                                              gap_score,
                                              mismatch_score,
                                              match_score,
                                              max_sequences_per_poa,
                                              sequence_begin_nodes_ids,
                                              outgoing_edges_coverage,
                                              outgoing_edges_coverage_count,
                                              max_nodes_per_graph,
                                              matrix_seq_dimension,
                                              batch_size.max_consensus_size,
                                              TPB);

    cudaPeekAtLastError();

    
    
    generateConsensusKernel<SizeT>
        <<<consensus_num_blocks, CUDAPOA_MAX_CONSENSUS_PER_BLOCK, 0, stream>>>(consensus_d,
                                                                                coverage_d,
                                                                                sequence_lengths_d,
                                                                                window_details_d,
                                                                                total_windows,
                                                                                nodes,
                                                                                incoming_edges,
                                                                                incoming_edge_count,
                                                                                outgoing_edges,
                                                                                outgoing_edge_count,
                                                                                incoming_edge_w,
                                                                                sorted_poa,
                                                                                node_id_to_pos,
                                                                                node_alignments,
                                                                                node_alignment_count,
                                                                                consensus_scores,
                                                                                consensus_predecessors,
                                                                                node_coverage_counts,
                                                                                max_nodes_per_graph,
                                                                                batch_size.max_consensus_size);
    cudaPeekAtLastError();
    
}