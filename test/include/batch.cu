#include <memory>

#include "batch.hpp"
#include "cudautility.hpp"
#include "cuda_limits.hpp"
#include "cuda_batch.cuh"

/// constructor- set other parameters based on a minimum set of input arguments
BatchConfig::BatchConfig(int32_t max_seq_sz /*= 1024*/, int32_t max_seq_per_poa /*= 100*/, int32_t band_width /*= 256*/,
                         BandMode banding /*= BandMode::full_band*/, float adapive_storage_factor /*= 2.0*/, float graph_length_factor /*= 3.0*/,
                         int32_t max_pred_dist /*= 0*/)
    /// ensure a 4-byte boundary alignment for any allocated buffer
    : max_sequence_size(max_seq_sz)
    , max_consensus_size(2 * max_sequence_size)
    /// ensure 128-alignment for band_width size, 128 = CUDAPOA_MIN_BAND_WIDTH
    , alignment_band_width(align<int32_t, CUDAPOA_MIN_BAND_WIDTH>(band_width))
    , max_sequences_per_poa(max_seq_per_poa)
    , band_mode(banding)
    , max_banded_pred_distance(max_pred_dist > 0 ? max_pred_dist : 2 * align<int32_t, CUDAPOA_MIN_BAND_WIDTH>(band_width))
{
    max_nodes_per_graph = align<int32_t, CUDAPOA_CELLS_PER_THREAD>(graph_length_factor * max_sequence_size);

    if (banding == BandMode::full_band)
    {
        matrix_sequence_dimension = align<int32_t, CUDAPOA_CELLS_PER_THREAD>(max_sequence_size);
    }
    else if (banding == BandMode::static_band || banding == BandMode::static_band_traceback)
    {
        matrix_sequence_dimension = align<int32_t, CUDAPOA_CELLS_PER_THREAD>(alignment_band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
    }
    else // BandMode::adaptive_band || BandMode::adaptive_band_traceback
    {
        // adapive_storage_factor is to reserve extra memory for cases with extended band-width
        matrix_sequence_dimension = align<int32_t, CUDAPOA_CELLS_PER_THREAD>(adapive_storage_factor * (alignment_band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING));
    }

    

    if (alignment_band_width != band_width)
    {
        std::cerr << "Band-width should be multiple of 128. The input was changed from " << band_width << " to " << alignment_band_width << std::endl;
    }
}

/// constructor- set all parameters separately
BatchConfig::BatchConfig(int32_t max_seq_sz, int32_t max_consensus_sz, int32_t max_nodes_per_poa, int32_t band_width,
                         int32_t max_seq_per_poa, int32_t matrix_seq_dim, BandMode banding, int32_t max_pred_distance)
    /// ensure a 4-byte boundary alignment for any allocated buffer
    : max_sequence_size(max_seq_sz)
    , max_consensus_size(max_consensus_sz)
    , max_nodes_per_graph(align<int32_t, CUDAPOA_CELLS_PER_THREAD>(max_nodes_per_poa))
    , matrix_sequence_dimension(align<int32_t, CUDAPOA_CELLS_PER_THREAD>(matrix_seq_dim))
    /// ensure 128-alignment for band_width size
    , alignment_band_width(align<int32_t, CUDAPOA_MIN_BAND_WIDTH>(band_width))
    , max_sequences_per_poa(max_seq_per_poa)
    , band_mode(banding)
    , max_banded_pred_distance(max_pred_distance)
{
    
    if (max_nodes_per_graph < max_sequence_size)
        throw std::invalid_argument("max_nodes_per_graph should be greater than or equal to max_sequence_size.");
    if (max_consensus_size < max_sequence_size)
        throw std::invalid_argument("max_consensus_size should be greater than or equal to max_sequence_size.");
    if (max_sequence_size < alignment_band_width)
        throw std::invalid_argument("alignment_band_width should not be greater than max_sequence_size.");
    if (alignment_band_width != band_width)
    {
        std::cerr << "Band-width should be multiple of 128. The input was changed from " << band_width << " to " << alignment_band_width << std::endl;
    }
}

std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    DefaultDeviceAllocator allocator,
                                    int64_t max_mem,
                                    int8_t output_mask,
                                    const BatchConfig& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score)
{
    if (use32bitScore(batch_size, gap_score, mismatch_score, match_score))
    {
        if (use32bitSize(batch_size))
        {
            if (use16bitTrace(batch_size))
            {
                return std::make_unique<CudapoaBatch<int32_t, int32_t, int16_t>>(device_id,
                                                                                 stream,
                                                                                 allocator,
                                                                                 max_mem,
                                                                                 output_mask,
                                                                                 batch_size,
                                                                                 gap_score,
                                                                                 mismatch_score,
                                                                                 match_score);
            }
            else
            {
                return std::make_unique<CudapoaBatch<int32_t, int32_t, int8_t>>(device_id,
                                                                                stream,
                                                                                allocator,
                                                                                max_mem,
                                                                                output_mask,
                                                                                batch_size,
                                                                                gap_score,
                                                                                mismatch_score,
                                                                                match_score);
            }
        }
        else
        {
            if (use16bitTrace(batch_size))
            {
                return std::make_unique<CudapoaBatch<int32_t, int16_t, int16_t>>(device_id,
                                                                                 stream,
                                                                                 allocator,
                                                                                 max_mem,
                                                                                 output_mask,
                                                                                 batch_size,
                                                                                 gap_score,
                                                                                 mismatch_score,
                                                                                 match_score);
            }
            else
            {
                return std::make_unique<CudapoaBatch<int32_t, int16_t, int8_t>>(device_id,
                                                                                stream,
                                                                                allocator,
                                                                                max_mem,
                                                                                output_mask,
                                                                                batch_size,
                                                                                gap_score,
                                                                                mismatch_score,
                                                                                match_score);
            }
        }
    }
    else
    {
        // if ScoreT is 16-bit, then it's safe to assume SizeT is 16-bit
        if (use16bitTrace(batch_size))
        {
            return std::make_unique<CudapoaBatch<int16_t, int16_t, int16_t>>(device_id,
                                                                             stream,
                                                                             allocator,
                                                                             max_mem,
                                                                             output_mask,
                                                                             batch_size,
                                                                             gap_score,
                                                                             mismatch_score,
                                                                             match_score);
        }
        else
        {
            return std::make_unique<CudapoaBatch<int16_t, int16_t, int8_t>>(device_id,
                                                                            stream,
                                                                            allocator,
                                                                            max_mem,
                                                                            output_mask,
                                                                            batch_size,
                                                                            gap_score,
                                                                            mismatch_score,
                                                                            match_score);
        }
    }
}

std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    int64_t max_mem,
                                    int8_t output_mask,
                                    const BatchConfig& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score)
{
    if (max_mem < -1)
    {
        throw std::invalid_argument("max_mem has to be either -1 (=all available GPU memory) or greater or equal than 0.");
    }
#ifdef GW_ENABLE_CACHING_ALLOCATOR
    // uses CachingDeviceAllocator
    if (max_mem == -1)
    {
        max_mem = find_largest_contiguous_device_memory_section();
        if (max_mem == 0)
        {
            throw std::runtime_error("No memory available for caching");
        }
    }
    DefaultDeviceAllocator allocator(max_mem);
#else
    // uses CudaMallocAllocator
    DefaultDeviceAllocator allocator;
#endif
    return create_batch(device_id, stream, allocator, max_mem, output_mask, batch_size, gap_score, mismatch_score, match_score);
}