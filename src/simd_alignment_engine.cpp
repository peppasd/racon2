#include "spoa/simd_alignment_engine.hpp"
#include "spoa/alignment_engine.hpp"
#include "spoa/graph.hpp"
#include "simdpp/simd.h"

namespace spoa
{
  const int SIMD_VECTOR_SIZE = 8;
  typedef simdpp::uint32<SIMD_VECTOR_SIZE> xuint;

  std::unique_ptr<AlignmentEngine> CreateSimdAlignmentEngine(
      std::int8_t m,
      std::int8_t n,
      std::int8_t g)
  {
    return SimdAlignmentEngine::Create(
        AlignmentType::Simd, m, n, g);
  }

  struct SimdAlignmentEngine::Implementation
  {
    std::vector<std::uint32_t> node_id_to_rank;
    std::vector<std::int32_t> sequence_profile;
    std::vector<std::int32_t> M;
    std::int32_t *H;

    Implementation()
        : node_id_to_rank(),
          sequence_profile(),
          M(),
          H(nullptr)
    {
    }
  };

  SimdAlignmentEngine::SimdAlignmentEngine(
      AlignmentType type,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g)
      : AlignmentEngine(type, m, n, g),
        pimpl_(new Implementation())
  {
  }

  std::unique_ptr<AlignmentEngine> SimdAlignmentEngine::Create(
      AlignmentType type,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g)
  {
    return std::unique_ptr<AlignmentEngine>(
        new SimdAlignmentEngine(type, m, n, g));
  }

  Alignment SimdAlignmentEngine::Align(
      const char *sequence, std::uint32_t sequence_len,
      const Graph &graph,
      std::int32_t *score)
  {
    if (sequence_len > std::numeric_limits<int32_t>::max())
    {
      throw std::invalid_argument(
          "[spoa::SimdAlignmentEngine::Align] error: too large sequence!");
    }

    if (graph.nodes().empty() || sequence_len == 0)
    {
      return Alignment();
    }

    std::uint32_t matrix_width = sequence_len + 1;
    std::uint32_t matrix_height = graph.nodes().size() + 1;
    std::uint8_t num_codes = graph.num_codes();

    pimpl_->node_id_to_rank.resize(matrix_height - 1, 0);
    pimpl_->sequence_profile.resize(num_codes * matrix_width, 0);
    pimpl_->M.resize(matrix_width * matrix_height, 0);
    pimpl_->H = pimpl_->M.data();

    for (std::uint32_t j = 1; j < matrix_width * matrix_height; j++)
    {
      pimpl_->H[j] = 0;
    }

    for (std::uint32_t i = 0; i < num_codes; ++i)
    {
      char c = graph.decoder(i);
      pimpl_->sequence_profile[i * matrix_width] = 0;
      for (std::uint32_t j = 0; j < sequence_len; ++j)
      {
        pimpl_->sequence_profile[i * matrix_width + (j + 1)] =
            (c == sequence[j] ? m_ : n_);
      }
    }

    const auto &rank_to_node = graph.rank_to_node();
    for (std::uint32_t i = 0; i < rank_to_node.size(); ++i)
    {
      pimpl_->node_id_to_rank[rank_to_node[i]->id] = i;
    }

    std::int32_t max_score = 0;
    std::uint32_t max_i = 0;
    std::uint32_t max_j = 0;
    auto update_max_score = [&max_score, &max_i, &max_j](
                                std::int32_t *H_row,
                                std::uint32_t i,
                                std::uint32_t j) -> void
    {
      if (max_score < H_row[j])
      {
        max_score = H_row[j];
        max_i = i;
        max_j = j;
      }
      return;
    };

    // alignment
    for (const auto &it : rank_to_node)
    {
      const auto &char_profile =
          &(pimpl_->sequence_profile[it->code * matrix_width]);

      std::uint32_t i = pimpl_->node_id_to_rank[it->id] + 1;
      std::uint32_t pred_i = it->inedges.empty() ? 0 : pimpl_->node_id_to_rank[it->inedges[0]->tail->id] + 1;

      std::int32_t *H_row = &(pimpl_->H[i * matrix_width]);
      std::int32_t *H_pred_row = &(pimpl_->H[pred_i * matrix_width]);

      // update H
      for (std::uint64_t j = 1; j < matrix_width; ++j)
      {
        H_row[j] = std::max(
            H_pred_row[j - 1] + char_profile[j],
            H_pred_row[j] + g_);
      }
      // check other predeccessors
      for (std::uint32_t p = 1; p < it->inedges.size(); ++p)
      {
        pred_i = pimpl_->node_id_to_rank[it->inedges[p]->tail->id] + 1;

        H_pred_row = &(pimpl_->H[pred_i * matrix_width]);

        for (std::uint64_t j = 1; j < matrix_width; ++j)
        {
          H_row[j] = std::max(
              H_pred_row[j - 1] + char_profile[j],
              std::max(
                  H_row[j],
                  H_pred_row[j] + g_));
        }
      }

      for (std::uint64_t j = 1; j < matrix_width; ++j)
      {
        H_row[j] = std::max(H_row[j - 1] + g_, H_row[j]);
        H_row[j] = std::max(H_row[j], 0);
        update_max_score(H_row, i, j);
      }
    }

    if (max_i == 0 && max_j == 0)
    {
      return Alignment();
    }
    if (score)
    {
      *score = max_score;
    }

    // backtrack
    Alignment alignment;
    std::uint32_t i = max_i;
    std::uint32_t j = max_j;

    std::uint32_t prev_i = 0;
    std::uint32_t prev_j = 0;

    while (pimpl_->H[i * matrix_width + j] == 0 ? false : true)
    {
      auto H_ij = pimpl_->H[i * matrix_width + j];
      bool predecessor_found = false;

      if (i != 0 && j != 0)
      {
        const auto &it = rank_to_node[i - 1];
        std::int32_t match_cost =
            pimpl_->sequence_profile[it->code * matrix_width + j];

        std::uint32_t pred_i = it->inedges.empty() ? 0 : pimpl_->node_id_to_rank[it->inedges[0]->tail->id] + 1;

        if (H_ij == pimpl_->H[pred_i * matrix_width + (j - 1)] + match_cost)
        {
          prev_i = pred_i;
          prev_j = j - 1;
          predecessor_found = true;
        }
        else
        {
          for (std::uint32_t p = 1; p < it->inedges.size(); ++p)
          {
            std::uint32_t pred_i =
                pimpl_->node_id_to_rank[it->inedges[p]->tail->id] + 1;

            if (H_ij == pimpl_->H[pred_i * matrix_width + (j - 1)] + match_cost)
            {
              prev_i = pred_i;
              prev_j = j - 1;
              predecessor_found = true;
              break;
            }
          }
        }
      }

      if (!predecessor_found && i != 0)
      {
        const auto &it = rank_to_node[i - 1];

        std::uint32_t pred_i = it->inedges.empty() ? 0 : pimpl_->node_id_to_rank[it->inedges[0]->tail->id] + 1;

        if (H_ij == pimpl_->H[pred_i * matrix_width + j] + g_)
        {
          prev_i = pred_i;
          prev_j = j;
          predecessor_found = true;
        }
        else
        {
          for (std::uint32_t p = 1; p < it->inedges.size(); ++p)
          {
            std::uint32_t pred_i =
                pimpl_->node_id_to_rank[it->inedges[p]->tail->id] + 1;

            if (H_ij == pimpl_->H[pred_i * matrix_width + j] + g_)
            {
              prev_i = pred_i;
              prev_j = j;
              predecessor_found = true;
              break;
            }
          }
        }
      }

      if (!predecessor_found && H_ij == pimpl_->H[i * matrix_width + j - 1] + g_)
      {
        prev_i = i;
        prev_j = j - 1;
        predecessor_found = true;
      }

      alignment.emplace_back(
          i == prev_i ? -1 : rank_to_node[i - 1]->id,
          j == prev_j ? -1 : j - 1);

      i = prev_i;
      j = prev_j;
    }

    std::reverse(alignment.begin(), alignment.end());
    return alignment;
  }
}