#pragma once

#include "allocate_block.hpp"
#include "batch.hpp"
#include "cudautility.hpp"
#include "cuda_kernel.cuh"
#include "signed_integer_utilits.hpp"
#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>

inline std::string printTabs(int32_t tab_count)
{
    std::string s;
    for (int32_t i = 0; i < tab_count; i++)
    {
        s += "\t";
    }
    return s;
}


template <typename ScoreT, typename SizeT, typename TraceT>
class CudapoaBatch : public Batch
{
public:
    CudapoaBatch(int32_t device_id, cudaStream_t stream, DefaultDeviceAllocator allocator, int64_t max_mem, int8_t output_mask,
                 const BatchConfig& batch_size, int32_t gap_score = -8, int32_t mismatch_score = -6, int32_t match_score = 8)
        : max_sequences_per_poa_(batch_size.max_sequences_per_poa)
        , device_id_(device_id)
        , stream_(stream)
        , output_mask_(output_mask)
        , batch_size_(batch_size)
        , gap_score_(gap_score)
        , mismatch_score_(mismatch_score)
        , match_score_(match_score)
        , batch_block_(new BatchBlock<ScoreT, SizeT, TraceT>(device_id,
                                                             allocator,
                                                             max_mem,
                                                             output_mask,
                                                             batch_size_))
        , max_poas_(batch_block_->get_max_poas())
    {
        // Set CUDA device
        scoped_device_switch dev(device_id_);

        bid_ = CudapoaBatch::batches++;

        std::string msg = " Initializing batch on device ";
        // print_batch_debug_message(msg);

        // Allocate host memory and CUDA memory based on max sequence and target counts.
        initialize_input_details();
        initialize_output_details();
        initialize_graph_details();
        initialize_alignment_details();

        // Call reset function to cleanly initialize members.
        reset();
    }

    ~CudapoaBatch()
    {
        std::string msg = " Destroyed buffers on device ";
        // print_batch_debug_message(msg);
    }

    virtual StatusType add_poa_group(std::vector<StatusType>& per_seq_status,
                                     const Group& poa_group)
    {
        // Check if the largest entry in the group fill fit
        // in available scoring matrix memory or not.
        auto max_length_entry  = std::max_element(poa_group.begin(),
                                                 poa_group.end(),
                                                 [](const Entry& lhs, const Entry& rhs) {
                                                     return lhs.length < rhs.length;
                                                 });
        int32_t max_seq_length = max_length_entry->length;

        //std::cout << "Adding new poa group!" << std::endl;

        if (!reserve_buf(max_seq_length))
        {
            return StatusType::exceeded_maximum_poas;
        }

        // If matrix fits, see if a new poa group can be added.
        per_seq_status.clear();
        StatusType status = add_poa();
        if (status != StatusType::success)
        {
            return status;
        }

        // If a new group can be added, attempt to add all entries
        // in the group. If they can't be added, record their status
        // and continue adding till the end of the group.
        bool poa_empty = true;
        for (auto& entry : poa_group)
        {
            StatusType entry_status = add_seq_to_poa(entry.seq,
                                                     entry.weights,
                                                     entry.length);
            if (entry_status == StatusType::success)
            {
                poa_empty = false;
            }
            per_seq_status.push_back(entry_status);
        }

        if (poa_empty)
        {
            return StatusType::empty_poa_group;
        }
        return StatusType::success;
    }

    // Get total number of partial order alignments in batch.
    int32_t get_total_poas() const
    {
        return poa_count_;
    }

    // Run partial order alignment algorithm over all POAs.
    void generate_poa()
    {
        scoped_device_switch dev(device_id_);

        if (poa_count_ == 0)
        {
            // print_batch_debug_message(" No POA was added to compute! ");
            return;
        }

        //Copy sequencecs, sequence lengths and window details to device
        cudaMemcpyAsync(input_details_d_->sequences, input_details_h_->sequences,
                                        num_nucleotides_copied_ * sizeof(*input_details_h_->sequences), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(input_details_d_->base_weights, input_details_h_->base_weights,
                                        num_nucleotides_copied_ * sizeof(*input_details_h_->base_weights), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(input_details_d_->window_details, input_details_h_->window_details,
                                        poa_count_ * sizeof(*input_details_h_->window_details), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(input_details_d_->sequence_lengths, input_details_h_->sequence_lengths,
                                        global_sequence_idx_ * sizeof(*input_details_h_->sequence_lengths), cudaMemcpyHostToDevice, stream_);

        // Launch kernel to run 1 POA per thread in thread block.
        std::string msg = " Launching kernel for " + std::to_string(poa_count_) + " on device ";
        // print_batch_debug_message(msg);

        generatePOA<ScoreT, SizeT, TraceT>(output_details_d_,
                                           input_details_d_,
                                           poa_count_,
                                           stream_,
                                           alignment_details_d_,
                                           graph_details_d_,
                                           gap_score_,
                                           mismatch_score_,
                                           match_score_,
                                           max_sequences_per_poa_,
                                           output_mask_,
                                           batch_size_);

        msg = " Launched kernel on device ";
        // print_batch_debug_message(msg);
    }

    // Get the consensus for each POA.
    StatusType get_consensus(std::vector<std::string>& consensus,
                             std::vector<std::vector<uint16_t>>& coverage,
                             std::vector<StatusType>& output_status)
    {
        // Check if consensus was requested at init time.
        if (!(OutputType::consensus & output_mask_))
        {
            return StatusType::output_type_unavailable;
        }

        std::string msg = " Launching memcpy D2H on device ";
        // print_batch_debug_message(msg);
        cudaMemcpyAsync(output_details_h_->consensus,
                                        output_details_d_->consensus,
                                        batch_size_.max_consensus_size * max_poas_ * sizeof(*output_details_h_->consensus),
                                        cudaMemcpyDeviceToHost,
                                        stream_);
        cudaMemcpyAsync(output_details_h_->coverage,
                                        output_details_d_->coverage,
                                        batch_size_.max_consensus_size * max_poas_ * sizeof(*output_details_h_->coverage),
                                        cudaMemcpyDeviceToHost,
                                        stream_);
        cudaStreamSynchronize(stream_);

        msg = " Finished memcpy D2H on device ";
        // print_batch_debug_message(msg);

        for (int32_t poa = 0; poa < poa_count_; poa++)
        {
            // Get the consensus string and reverse it since on GPU the
            // string is built backwards..
            char* c = reinterpret_cast<char*>(&(output_details_h_->consensus[poa * batch_size_.max_consensus_size]));
            // We use the first two entries in the consensus buffer to log error during kernel execution
            // c[0] == 0 means an error occured and when that happens the error type is saved in c[1]
            if (static_cast<uint8_t>(c[0]) == CUDAPOA_KERNEL_ERROR_ENCOUNTERED)
            {
                // decode_cudapoa_kernel_error(static_cast<genomeworks::cudapoa::StatusType>(c[1]), output_status);
                // push back empty placeholder for consensus and coverage
                consensus.emplace_back(std::string());
                coverage.emplace_back(std::vector<uint16_t>());
            }
            else
            {
                output_status.emplace_back(StatusType::success);
                consensus.emplace_back(std::string(c));
                std::reverse(consensus.back().begin(), consensus.back().end());
                // Similarly, get the coverage and reverse it.
                coverage.emplace_back(std::vector<uint16_t>(
                    &(output_details_h_->coverage[poa * batch_size_.max_consensus_size]),
                    &(output_details_h_->coverage[poa * batch_size_.max_consensus_size + get_size(consensus.back())])));
                std::reverse(coverage.back().begin(), coverage.back().end());
                //std::cout << consensus.back() << std::endl;
            }
        }

        return StatusType::success;
    }

    // Get multiple sequence alignments for each POA
    StatusType get_msa(std::vector<std::vector<std::string>>& msa,
                       std::vector<StatusType>& output_status)
    {
        // Check if msa was requested at init time.
        if (!(OutputType::msa & output_mask_))
        {
            return StatusType::output_type_unavailable;
        }

        std::string msg = " Launching memcpy D2H on device for msa ";
        // print_batch_debug_message(msg);

        cudaMemcpyAsync(output_details_h_->multiple_sequence_alignments,
                                        output_details_d_->multiple_sequence_alignments,
                                        max_poas_ * max_sequences_per_poa_ * batch_size_.max_consensus_size * sizeof(*output_details_h_->multiple_sequence_alignments),
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        cudaMemcpyAsync(output_details_h_->consensus,
                                        output_details_d_->consensus,
                                        batch_size_.max_consensus_size * max_poas_ * sizeof(*output_details_h_->consensus),
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        cudaStreamSynchronize(stream_);

        msg = " Finished memcpy D2H on device for msa";
        // print_batch_debug_message(msg);

        for (int32_t poa = 0; poa < poa_count_; poa++)
        {
            msa.emplace_back(std::vector<std::string>());
            char* c = reinterpret_cast<char*>(&(output_details_h_->consensus[poa * batch_size_.max_consensus_size]));
            // We use the first two entries in the consensus buffer to log error during kernel execution
            // c[0] == 0 means an error occured and when that happens the error type is saved in c[1]
            if (static_cast<uint8_t>(c[0]) == CUDAPOA_KERNEL_ERROR_ENCOUNTERED)
            {
                // decode_cudapoa_kernel_error(static_cast<genomeworks::cudapoa::StatusType>(c[1]), output_status);
                std::cout << "CUDAPOA_KERNEL_ERROR_ENCOUNTERED\n";
            }
            else
            {
                output_status.emplace_back(StatusType::success);
                uint16_t num_seqs = input_details_h_->window_details[poa].num_seqs;
                for (int32_t i = 0; i < num_seqs; i++)
                {
                    char* c = reinterpret_cast<char*>(&(output_details_h_->multiple_sequence_alignments[(poa * max_sequences_per_poa_ + i) * batch_size_.max_consensus_size]));
                    msa[poa].emplace_back(std::string(c));
                }
            }
        }

        return StatusType::success;
    }

    void get_graphs(std::vector<DirectedGraph>& graphs,
                    std::vector<StatusType>& output_status)
    {
        int32_t max_nodes_per_window_ = batch_size_.max_nodes_per_graph;
        cudaMemcpyAsync(graph_details_h_->nodes,
                                        graph_details_d_->nodes,
                                        sizeof(*graph_details_h_->nodes) * max_nodes_per_window_ * max_poas_,
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        cudaMemcpyAsync(graph_details_h_->incoming_edges,
                                        graph_details_d_->incoming_edges,
                                        sizeof(*graph_details_h_->incoming_edges) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_,
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        cudaMemcpyAsync(graph_details_h_->incoming_edge_weights,
                                        graph_details_d_->incoming_edge_weights,
                                        sizeof(*graph_details_h_->incoming_edge_weights) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_,
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        cudaMemcpyAsync(graph_details_h_->incoming_edge_count,
                                        graph_details_d_->incoming_edge_count,
                                        sizeof(*graph_details_h_->incoming_edge_count) * max_nodes_per_window_ * max_poas_,
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        cudaMemcpyAsync(input_details_h_->sequence_lengths,
                                        input_details_d_->sequence_lengths,
                                        global_sequence_idx_ * sizeof(*input_details_h_->sequence_lengths),
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        cudaMemcpyAsync(output_details_h_->consensus,
                                        output_details_d_->consensus,
                                        batch_size_.max_consensus_size * max_poas_ * sizeof(*output_details_h_->consensus),
                                        cudaMemcpyDeviceToHost,
                                        stream_);

        // Reservet host space for graphs
        graphs.resize(poa_count_);

        cudaStreamSynchronize(stream_);

        for (int32_t poa = 0; poa < poa_count_; poa++)
        {
            char* c = reinterpret_cast<char*>(&(output_details_h_->consensus[poa * batch_size_.max_consensus_size]));
            // We use the first two entries in the consensus buffer to log error during kernel execution
            // c[0] == 0 means an error occured and when that happens the error type is saved in c[1]
            if (static_cast<uint8_t>(c[0]) == CUDAPOA_KERNEL_ERROR_ENCOUNTERED)
            {
                // decode_cudapoa_kernel_error(static_cast<genomeworks::cudapoa::StatusType>(c[1]), output_status);
                std::cout << "CUDAPOA_KERNEL_ERROR_ENCOUNTERED\n";
            }
            else
            {
                output_status.emplace_back(StatusType::success);
                DirectedGraph& graph = graphs[poa];
                int32_t seq_0_offset = input_details_h_->window_details[poa].seq_len_buffer_offset;
                int32_t num_nodes    = input_details_h_->sequence_lengths[seq_0_offset];
                uint8_t* nodes       = &graph_details_h_->nodes[max_nodes_per_window_ * poa];
                for (int32_t n = 0; n < num_nodes; n++)
                {
                    // For each node, find it's incoming edges and add the edge to the graph,
                    // along with its label.
                    DirectedGraph::node_id_t sink = n;
                    graph.set_node_label(sink, std::string(1, static_cast<char>(nodes[n])));
                    uint16_t num_edges = graph_details_h_->incoming_edge_count[poa * max_nodes_per_window_ + n];
                    for (int32_t e = 0; e < num_edges; e++)
                    {
                        int32_t idx                         = poa * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES + n * CUDAPOA_MAX_NODE_EDGES + e;
                        DirectedGraph::node_id_t src        = graph_details_h_->incoming_edges[idx];
                        DirectedGraph::edge_weight_t weight = graph_details_h_->incoming_edge_weights[idx];
                        graph.add_edge(src, sink, weight);
                    }
                }
            }
        }
    }

    // Return batch ID.
    int32_t batch_id() const
    {
        return bid_;
    }

    // Reset batch. Must do before re-using batch.
    void reset()
    {
        poa_count_              = 0;
        num_nucleotides_copied_ = 0;
        global_sequence_idx_    = 0;
        next_scores_offset_     = 0;
        avail_buf_mem_          = alignment_details_d_->scorebuf_alloc_size;
    }

protected:
    // Print debug message with batch specific formatting.
    // void print_batch_debug_message(const std::string& message)
    // {
    //     std::string msg = printTabs(bid_) + " " + std::to_string(bid_) + " " + message + " " + std::to_string(device_id_);
    //     GW_LOG_DEBUG(msg.c_str());
    // }

    // Allocate buffers for output details
    void initialize_output_details()
    {
        batch_block_->get_output_details(&output_details_h_, &output_details_d_);
    }

    // Allocate buffers for alignment details
    void initialize_alignment_details()
    {
        batch_block_->get_alignment_details(&alignment_details_d_);
    }

    // Allocate buffers for graph details
    void initialize_graph_details()
    {
        batch_block_->get_graph_details(&graph_details_d_, &graph_details_h_);
    }

    // Allocate buffers for input details
    void initialize_input_details()
    {
        batch_block_->get_input_details(&input_details_h_, &input_details_d_);
    }

    // Log cudapoa kernel error
    // void decode_cudapoa_kernel_error(genomeworks::cudapoa::StatusType error_type,
    //                                  std::vector<StatusType>& output_status)
    // {
    //     std::string error_message;
    //     std::string error_hint;
    //     decode_error(error_type, error_message, error_hint);
    //     error_message += " in batch " + std::to_string(bid_) + "\n" + error_hint;
    //     GW_LOG_WARN(error_message.c_str());
    //     output_status.emplace_back(error_type);
    // }

    // Add new partial order alignment to batch.
    StatusType add_poa()
    {
        if (poa_count_ == max_poas_)
        {
            return StatusType::exceeded_maximum_poas;
        }

        WindowDetails window_details{};
        window_details.seq_len_buffer_offset         = global_sequence_idx_;
        window_details.seq_starts                    = num_nucleotides_copied_;
        window_details.scores_width                  = 0;
        window_details.scores_offset                 = next_scores_offset_;
        input_details_h_->window_details[poa_count_] = window_details;
        poa_count_++;

        return StatusType::success;
    }

    // Add sequence to last partial order alignment.
    StatusType add_seq_to_poa(const char* seq, const int8_t* weights, int32_t seq_len)
    {
        if (seq_len > (int32_t)batch_size_.max_sequence_size)
        {
            return StatusType::exceeded_maximum_sequence_size;
        }

        if (weights != nullptr)
        {
            // Verify that weights are positive.
            bool all_base_weights_are_zero = true;
            for (int32_t i = 0; i < seq_len; i++)
            {
                if (weights[i] > 0)
                {
                    all_base_weights_are_zero = false;
                }
            }
            // all base weights of the sequence can not be zero, skip
            if (all_base_weights_are_zero)
            {
                return StatusType::zero_weighted_poa_sequence;
            }
        }

        WindowDetails* window_details = &(input_details_h_->window_details[poa_count_ - 1]);
        int32_t scores_width_         = align<int32_t, 4>(seq_len + 1 + CUDAPOA_CELLS_PER_THREAD);
        if (scores_width_ > window_details->scores_width)
        {
            next_scores_offset_ += (scores_width_ - window_details->scores_width);
            window_details->scores_width = scores_width_;
        }

        if (static_cast<int32_t>(window_details->num_seqs) >= max_sequences_per_poa_)
        {
            return StatusType::exceeded_maximum_sequences_per_poa;
        }

        window_details->num_seqs++;
        // Copy sequence data
        memcpy(&(input_details_h_->sequences[num_nucleotides_copied_]),
               seq,
               seq_len);
        // Copy weights
        if (weights == nullptr)
        {
            memset(&(input_details_h_->base_weights[num_nucleotides_copied_]),
                   1,
                   seq_len);
        }
        else
        {
            memcpy(&(input_details_h_->base_weights[num_nucleotides_copied_]),
                   weights,
                   seq_len);
        }
        input_details_h_->sequence_lengths[global_sequence_idx_] = seq_len;

        // to be aligned with SeqT4 struct size, pad sequence length to be multiple of 4.
        // Note: num_nucleotides_copied_ is used to define allocated space per read on device as well as on host,
        // therefore it is important to reflect any changes in the following line in the corresponding device code as well
        num_nucleotides_copied_ += align<int32_t, SIZE_OF_SeqT4>(seq_len);

        global_sequence_idx_++;

        return StatusType::success;
    }

    // Check if intermediate data for seq length can fit in available scoring/traceback buffer
    bool reserve_buf(int32_t max_seq_length)
    {
        int32_t matrix_height = batch_size_.max_nodes_per_graph;
        // matrix width for full_band is based on the current group max_seq_length as opposed to batch_size_.matrix_sequence_dimension.
        // The latter is based on the largest group in the batch and is more conservative
        int32_t matrix_width = (batch_size_.band_mode != BandMode::full_band) ? batch_size_.matrix_sequence_dimension : align<int32_t, 4>(max_seq_length + 1 + CUDAPOA_CELLS_PER_THREAD);
        // in traceback alignments avail_buf_mem_ is dedicated to traceback matrix, otherwise it is being used for score matrix
        size_t required_size = static_cast<size_t>(matrix_width) * static_cast<size_t>(matrix_height);
        required_size *= (batch_size_.band_mode == static_band_traceback || batch_size_.band_mode == adaptive_band_traceback) ? sizeof(TraceT) : sizeof(ScoreT);

        if (required_size > avail_buf_mem_)
        {
            if (get_total_poas() == 0)
            {
                std::cout << "Memory available " << std::fixed << std::setprecision(2) << (static_cast<double>(avail_buf_mem_)) / 1024. / 1024. / 1024.;
                std::cout << "GB, Memory required " << (static_cast<double>(required_size)) / 1024. / 1024. / 1024.;
                std::cout << "GB (sequence length " << max_seq_length << ", graph length " << matrix_height << ")" << std::endl;
            }
            return false;
        }
        else
        {
            avail_buf_mem_ -= required_size;
            return true;
        }
    }

protected:
    // Maximum sequences per POA.
    int32_t max_sequences_per_poa_ = 0;

    // GPU Device ID
    int32_t device_id_ = 0;

    // CUDA stream for launching kernels.
    cudaStream_t stream_;

    // Bit field for output type
    int8_t output_mask_;

    // Upper limits for data size
    BatchConfig batch_size_;

    // Gap, mismatch and match scores for NW dynamic programming loop.
    int32_t gap_score_;
    int32_t mismatch_score_;
    int32_t match_score_;

    // Host and device buffer for output data.
    OutputDetails* output_details_h_;
    OutputDetails* output_details_d_;

    // Host and device buffer pointer for input data.
    InputDetails<SizeT>* input_details_d_;
    InputDetails<SizeT>* input_details_h_;

    // Device buffer struct for alignment details
    AlignmentDetails<ScoreT, SizeT, TraceT>* alignment_details_d_;

    // Device buffer struct for graph details
    GraphDetails<SizeT>* graph_details_d_;
    GraphDetails<SizeT>* graph_details_h_;

    // Batch ID.
    int32_t bid_ = 0;

    // Total POAs added.
    int32_t poa_count_ = 0;

    // Number of nucleotides already already inserted.
    int32_t num_nucleotides_copied_ = 0;

    // Global sequence index.
    int32_t global_sequence_idx_ = 0;

    // Remaining buffer memory available for scores matrices in case of full alignment, and for traceback matrices in case of banded alignment
    size_t avail_buf_mem_ = 0;

    // Temporary variable to compute the offset to scorebuf.
    size_t next_scores_offset_ = 0;

    // flag that enables some extra buffers to accommodate fully adaptive bands with variable width and arbitrary location
    // disabled for current implementation, can be enabled for possible future variants of adaptive alignment algorithm
    bool variable_band_ = false;

    // Pointer of a seperate class BatchBlock that implements details on calculating and allocating the memory for each batch
    std::unique_ptr<BatchBlock<ScoreT, SizeT, TraceT>> batch_block_;

    // Maximum POAs to process in batch.
    int32_t max_poas_ = 0;

public:
    // Static batch count used to generate batch IDs.
    static int32_t batches;
};

template <typename ScoreT, typename SizeT, typename TraceT>
int32_t CudapoaBatch<ScoreT, SizeT, TraceT>::batches = 0;