#include "spoa/simd_alignment_engine.hpp"
#include "spoa/alignment_engine.hpp"
#include "spoa/graph.hpp"
#include "simdpp/simd.h"

namespace spoa
{
  const int SIMD_VECTOR_SIZE = 8;
  typedef simdpp::int32<SIMD_VECTOR_SIZE> xint32;

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

    std::int32_t kNegativeInfinity = std::numeric_limits<std::int32_t>::min() + 1024;
    // SIMD padded matrix_width
    std::uint32_t matrix_width = sequence_len + 1 + (SIMD_VECTOR_SIZE - (sequence_len + 1) % SIMD_VECTOR_SIZE);
    std::uint32_t vector_matrix_width = matrix_width / SIMD_VECTOR_SIZE;
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

      for (std::uint32_t j = 0; j < vector_matrix_width; ++j)
      {
        for (std::uint32_t k = 0; k < SIMD_VECTOR_SIZE; ++k)
        {
          pimpl_->sequence_profile[i * matrix_width + (j * SIMD_VECTOR_SIZE + k)] =
              j * SIMD_VECTOR_SIZE + k < sequence_len ? (c == sequence[j] ? m_ : n_) : -1;
        }
      }
    }

    const auto &rank_to_node = graph.rank_to_node();
    for (std::uint32_t i = 0; i < rank_to_node.size(); ++i)
    {
      pimpl_->node_id_to_rank[rank_to_node[i]->id] = i;
    }

    std::int32_t max_score = kNegativeInfinity;
    std::uint32_t max_i = -1;
    std::uint32_t max_j = -1;

    // alignment
    for (const auto &it : rank_to_node)
    {
      const auto &char_profile =
          &(pimpl_->sequence_profile[it->code * matrix_width]);

      std::uint32_t i = pimpl_->node_id_to_rank[it->id] + 1;
      std::uint32_t pred_i = it->inedges.empty() ? 0 : pimpl_->node_id_to_rank[it->inedges[0]->tail->id] + 1;

      std::int32_t *H_row = &(pimpl_->H[i * matrix_width]);
      std::int32_t *H_pred_row = &(pimpl_->H[pred_i * matrix_width]);

      int32_t fill_array[SIMD_VECTOR_SIZE];
      std::fill_n(fill_array, SIMD_VECTOR_SIZE, g_);
      xint32 xg = simdpp::load(fill_array);

      std::fill_n(fill_array, SIMD_VECTOR_SIZE, 0);
      xint32 xZero = simdpp::load(fill_array);

      xint32 x = xZero;
      // update H
      for (std::uint32_t vector_j = 1; vector_j < vector_matrix_width; ++vector_j)
      {
        std::uint32_t j = vector_j * SIMD_VECTOR_SIZE;

        xint32 xChar_Profile = simdpp::load(char_profile + j);
        xint32 xH_pred_row = simdpp::load(H_pred_row + j);
        xint32 xH_row = (xH_pred_row << 4) | x;

        x = xH_pred_row >> 28;

        xH_row = simdpp::max(xH_row + xChar_Profile, xH_pred_row + xg);

        simdpp::store(H_row + j, xH_row);
      }
      // check other predeccessors
      for (std::uint32_t p = 1; p < it->inedges.size(); ++p)
      {
        pred_i = pimpl_->node_id_to_rank[it->inedges[p]->tail->id] + 1;

        H_pred_row = &(pimpl_->H[pred_i * matrix_width]);

        x = xZero;

        for (std::uint32_t vector_j = 1; vector_j < vector_matrix_width; ++vector_j)
        {
          std::uint32_t j = vector_j * SIMD_VECTOR_SIZE;

          xint32 xChar_Profile = simdpp::load(char_profile + j);
          xint32 xH_pred_row = simdpp::load(H_pred_row + j);
          xint32 xH_row = simdpp::load(H_row + j);
          xint32 xM = (xH_pred_row << 4) | x;

          x = xH_pred_row >> 28;

          xH_row = simdpp::max(xH_row, simdpp::max(xM + xChar_Profile, xH_pred_row + xg));

          simdpp::store(H_row + j, xH_row);
        }
      }
      std::fill_n(fill_array, SIMD_VECTOR_SIZE, kNegativeInfinity);
      xint32 score = simdpp::load(fill_array);
      x = xg >> 28;

      for (std::uint32_t vector_j = 1; vector_j < vector_matrix_width; ++vector_j)
      {
        std::uint32_t j = vector_j * SIMD_VECTOR_SIZE;
        xint32 xH_row = simdpp::load(H_row + j);

        // TODO masks
        xH_row = simdpp::max(xH_row, x);

        // TODO prefix max
        x = (xH_row + xg) >> 28;

        xH_row = simdpp::max(xH_row, xZero);
        score = simdpp::max(score, xH_row);

        simdpp::store(H_row + j, xH_row);
      }

      std::int32_t max_row_score = simdpp::reduce_max(score);
      if (max_score < max_row_score)
      {
        max_score = max_row_score;
        max_i = i;
      }
    }

    if (max_i == -1 && max_j == -1)
    {
      return Alignment();
    }
    if (score)
    {
      *score = max_score;
    }

    std::int32_t *H_row = &(pimpl_->H[max_i * matrix_width]);
    for (std::uint32_t j = 1; j < matrix_width; ++j)
    {
      if (H_row[j] == max_score)
      {
        max_j = j;
        break;
      }
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