#ifndef CUDA_ALIGNMENT_ENGINE_HPP_
#define CUDA_ALIGNMENT_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "spoa/alignment_engine.hpp"

namespace spoa
{

  std::unique_ptr<AlignmentEngine> CreateCudaAlignmentEngine(
      std::int8_t m,
      std::int8_t n,
      std::int8_t g);
} // namespace spoa

#endif // CUDA_ALIGNMENT_ENGINE_HPP_