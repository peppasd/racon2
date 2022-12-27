#include "spoa/simd_alignment_engine.hpp"
#include "spoa/alignment_engine.hpp"
#include "spoa/graph.hpp"
#include "simdpp/simd.h"

namespace spoa
{
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

    std::unique_ptr<simdpp::uint32<8>> sequence_profile_storage;
    std::uint64_t sequence_profile_size;
    simdpp::uint32<8> *sequence_profile;

    std::vector<std::int32_t> first_column;
    std::unique_ptr<simdpp::uint32<8>> M_storage;
    std::uint64_t M_size;
    simdpp::uint32<8> *H;

    std::unique_ptr<simdpp::uint32<8>> masks_storage;
    std::uint32_t masks_size;
    simdpp::uint32<8> *masks;

    std::unique_ptr<simdpp::uint32<8>> penalties_storage;
    std::uint32_t penalties_size;
    simdpp::uint32<8> *penalties;

    Implementation()
        : node_id_to_rank(),
          sequence_profile_storage(nullptr),
          sequence_profile_size(0),
          sequence_profile(nullptr),
          first_column(),
          M_storage(nullptr),
          M_size(0),
          H(nullptr),
          masks_storage(nullptr),
          masks_size(0),
          masks(nullptr),
          penalties_storage(nullptr),
          penalties_size(0),
          penalties(nullptr)
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

    // Realloc

    // Initialize

    // Linear

    return Alignment();
  }
}