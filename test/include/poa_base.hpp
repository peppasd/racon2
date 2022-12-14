#pragma once

#include <string>


enum StatusType
{
    success = 0,
    exceeded_maximum_poas,
    exceeded_maximum_sequence_size,
    exceeded_maximum_sequences_per_poa,
    node_count_exceeded_maximum_graph_size,
    edge_count_exceeded_maximum_graph_size,
    exceeded_adaptive_banded_matrix_size,
    exceeded_maximum_predecessor_distance,
    loop_count_exceeded_upper_bound,
    output_type_unavailable,
    zero_weighted_poa_sequence,
    empty_poa_group,
    generic_error
};



/// Generate corresponding error message for a given error type
/// \param [in] error_type input error code
/// \param [out] error_message corresponding error message
/// \param [out] error_hint possible hint to resolve the error
void decode_error(StatusType error_type, std::string& error_message, std::string& error_hint);



/// Banding mode used in Needleman-Wunsch algorithm
/// - full_band performs computations on full scores matrix, highest accuracy
/// - static_band performs computations on a fixed band along scores matrix diagonal, fastest implementation
/// - adaptive_band, similar to static_band performs computations on a band along diagonal, but the band-width
///   can vary per alignment's score matrix, faster than full_band and more accurate than static_band
/// - static_band_traceback is similar to static_band, but uses traceback matrix. In this mode, score matrix is only
///   partially stored. The height of score matrix is equivalent to maximum predecessors distance and this maximum
///   distance is limited and smaller than full POA graph length, this can be a source of difference vs static_band.
///   Traceback matrix requires less memory compared to score matrix, and this banding mode can be useful for
///   long-read cases where GPU memory is limiting the parallelism.
/// - adaptive_band_traceback, similar to static_band_traceback but with varying band-width size
enum BandMode
{
    full_band = 0,
    static_band,
    adaptive_band,
    static_band_traceback,
    adaptive_band_traceback
};


/// Initialize CUDA POA context.
StatusType Init();

/// OutputType - Enum for encoding type of output
enum OutputType
{
    consensus = 0x1,
    msa       = 0x1 << 1
};