#include "spoa/alignment_engine.hpp"
#include "spoa/simd_alignment_engine.hpp"
// #include "spoa/cuda_alignment_engine.hpp"
// #include "spoa/multithread_alignment_engine.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>

namespace spoa
{

  std::unique_ptr<AlignmentEngine> AlignmentEngine::Create(
      AlignmentType type,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g)
  {
    if (g > 0)
    {
      throw std::invalid_argument(
          "[spoa::AlignmentEngine::Create] error: "
          "gap opening penalty must be non-positive!");
    }

    auto dst = CreateSimdAlignmentEngine(m, n, g);
    // switch (type)
    // {
    // case AlignmentType::Simd:
    //   dst = CreateSimdAlignmentEngine(m, n, g);
    //   break;
    // case AlignmentType::Cuda:
    //   dst = CreateCudaAlignmentEngine(m, n, g);
    //   break;
    // case AlignmentType::Multithread:
    //   dst = CreateMultithreadAlignmentEngine(m, n, g);
    //   break;
    // default:
    //   throw std::invalid_argument(
    //       "[spoa::AlignmentEngine::Create] error: "
    //       "invalid alignment type!");
    //   break;
    // }
    return dst;
  }

  AlignmentEngine::AlignmentEngine(
      AlignmentType type,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g)
      : type_(type),
        m_(m),
        n_(n),
        g_(g)
  {
  }

  Alignment AlignmentEngine::Align(
      const std::string &sequence,
      const Graph &graph,
      std::int32_t *score)
  {
    return Align(sequence.c_str(), sequence.size(), graph, score);
  }

  // std::int64_t AlignmentEngine::WorstCaseAlignmentScore(
  //     std::int64_t i,
  //     std::int64_t j) const
  // {
  //   auto gap_score = [&](std::int64_t len) -> std::int64_t
  //   {
  //     return len == 0 ? 0 : std::min(g_ + (len - 1) * e_, q_ + (len - 1) * c_);
  //   };
  //   return std::min(
  //       -1 * (m_ * std::min(i, j) + gap_score(std::abs(i - j))),
  //       gap_score(i) + gap_score(j));
  // }

} // namespace spoa
