
#pragma once

#include "batch.hpp"
#include "signed_integer_utilits.hpp"

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cassert>



/// \brief Create a small set of batch-sizes to increase GPU parallelization.
///        Input POA groups are binned based on their sizes. Similarly sized poa_groups are grouped in the same bin.
///        Separating smaller POA groups from larger ones allows to launch a larger number of groups concurrently.
///        This increase in parallelization translates to faster runtime
///        This multi-batch strategy is particularly useful when input POA groups sizes display a large variance.
///
/// \param list_of_batch_sizes [out]        a set of batch-sizes, covering all input poa_groups
/// \param list_of_groups_per_batch [out]   corresponding POA groups per batch-size bin
/// \param poa_groups [in]                  vector of input poa_groups
/// \param msa_flag [in]                    flag indicating whether MSA or consensus is going to be computed, default is consensus
/// \param band_width [in]                  band-width used in static band mode, it also defines minimum band-width in adaptive band mode
/// \param band_mode [in]                   defining which banding mod is selected: full , static or adaptive
/// \param bins_capacity [in]               pointer to vector of bins used to create separate different-sized poa_groups, if null as input, a set of default bins will be used
/// \param gpu_memory_usage_quota [in]      portion of GPU available memory that will be used for compute each cudaPOA batch, default 0.9
/// \param mismatch_score [in]              mismatch score, default -6
/// \param gap_score [in]                   gap score, default -8
/// \param match_score [in]                 match core, default 8
void get_multi_batch_sizes(std::vector<BatchConfig>& list_of_batch_sizes,
                           std::vector<std::vector<int32_t>>& list_of_groups_per_batch,
                           const std::vector<Group>& poa_groups,
                           bool msa_flag                       = false,
                           int32_t band_width                  = 256,
                           BandMode band_mode                  = BandMode::adaptive_band,
                           float adaptive_storage_factor       = 2.0f,
                           float graph_length_factor           = 3.0f,
                           int32_t max_pred_distance           = 0,
                           std::vector<int32_t>* bins_capacity = nullptr,
                           float gpu_memory_usage_quota        = 0.9,
                           int32_t mismatch_score              = -6,
                           int32_t gap_score                   = -8,
                           int32_t match_score                 = 8);

/// \brief Resizes input windows to specified size in total_windows if total_windows >= 0
///
/// \param[out] windows      Reference to vector into which parsed window
///                          data is saved
/// \param[in] total_windows Limit windows read to total windows, or
///                          loop over existing windows to fill remaining spots.
///                          -1 ignores the total_windows arg and uses all windows in the file.
inline void resize_windows(std::vector<std::vector<std::string>>& windows, const int32_t total_windows)
{
    if (total_windows >= 0)
    {
        if (get_size(windows) > total_windows)
        {
            windows.erase(windows.begin() + total_windows, windows.end());
        }
        else if (get_size(windows) < total_windows)
        {
            int32_t windows_read = windows.size();
            while (get_size(windows) != total_windows)
            {
                windows.push_back(windows[windows.size() - windows_read]);
            }
        }

        assert(get_size(windows) == total_windows);
    }
}

/// \brief Parses cudapoa data file in the following format:
///        <num_sequences_in_window_0>
///        window0_seq0
///        window0_seq1
///        window0_seq2
///        ...
///        ...
///        <num_sequences_in_window_1>
///        window1_seq0
///        window1_seq1
///        window1_seq2
///        ...
///        ...
/// \param[out] windows Reference to vector into which parsed window
///                     data is saved
/// \param[in] filename Name of file with window data
/// \param[in] total_windows Limit windows read to total windows, or
///                          loop over existing windows to fill remaining spots.
///                          -1 ignored the total_windows arg and uses all windows in the file.
inline void parse_cudapoa_file(std::vector<std::vector<std::string>>& windows, const std::string& filename, int32_t total_windows)
{
    std::ifstream infile(filename);
    std::string line;
    int32_t windows_count = -1;
    while(getline(infile, line)){
        if((line[0]-'0') > 0 && (line[0]-'9')<0){
            windows.emplace_back(std::vector<std::string>());
            ++windows_count; 
        }
        else{
            windows[windows_count].push_back(line);
        }
    }
    resize_windows(windows, total_windows);
}

/// \brief Parses windows from 1 or more fasta files
///
/// \param[out] windows Reference to vector into which parsed window
///                     data is saved
/// \param[in] input_filepaths Reference to vector containing names of fasta files with window data
/// \param[in] total_windows Limit windows read to total windows, or
///                          loop over existing windows to fill remaining spots.
///                          -1 ignored the total_windows arg and uses all windows in the file.
// inline void parse_fasta_files(std::vector<std::vector<std::string>>& windows, const std::vector<std::string>& input_paths, const int32_t total_windows)
// {
//     std::ifstream infile(filename);
//     string line;
//     int32_t windows_count = -1;
//     while(getline(infile, line)){
//         if((line[0]-'0') > 0 && (line[0]-'9')<0){
//             windows.emplace_back(vector<string>());
//             ++windows_count; 
//         }
//         else{
//             windows[windows_count].push_back(line);
//         }
//     }
//     resize_windows(windows, total_windows);
// }

/// \brief Parses golden value file with genome
///
/// \param[in] filename Name of file with reference genome
///
/// \return Genome string
// inline std::string parse_golden_value_file(const std::string& filename)
// {
//     std::ifstream infile(filename);
//     if (!infile.good())
//     {
//         throw std::runtime_error("Cannot read file " + filename);
//     }

//     std::string line;
//     std::getline(infile, line);
//     return line;
// }
