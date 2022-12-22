#ifndef MULTITHREAD_ALIGNMENT_ENGINE_HPP_
#define MULTITHREAD_ALIGNMENT_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "spoa/alignment_engine.hpp"

namespace spoa
{

  std::unique_ptr<AlignmentEngine> CreateMultithreadAlignmentEngine(
      std::int8_t m,
      std::int8_t n,
      std::int8_t g);
} // namespace spoa

#endif // SIMD_ALIGNMENT_ENGINE_HPP_