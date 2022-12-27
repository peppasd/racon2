#ifndef SIMD_ALIGNMENT_ENGINE_HPP_
#define SIMD_ALIGNMENT_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "spoa/alignment_engine.hpp"

namespace spoa
{

  std::unique_ptr<AlignmentEngine> CreateSimdAlignmentEngine(
      std::int8_t m,
      std::int8_t n,
      std::int8_t g);

  class SimdAlignmentEngine : public AlignmentEngine
  {
  public:
    SimdAlignmentEngine(const SimdAlignmentEngine &) = delete;
    SimdAlignmentEngine &operator=(const SimdAlignmentEngine &) = delete;

    SimdAlignmentEngine(SimdAlignmentEngine &&) = default;
    SimdAlignmentEngine &operator=(SimdAlignmentEngine &&) = delete;

    ~SimdAlignmentEngine() = default;

    static std::unique_ptr<AlignmentEngine> Create(
        AlignmentType type,
        std::int8_t m,
        std::int8_t n,
        std::int8_t g);

    void Prealloc(
        std::uint32_t max_sequence_len,
        std::uint8_t alphabet_size) override;

    Alignment Align(
        const char *sequence, std::uint32_t sequence_len,
        const Graph &graph,
        std::int32_t *score) override;

    friend std::unique_ptr<AlignmentEngine> CreateSimdAlignmentEngine(
        std::int8_t m,
        std::int8_t n,
        std::int8_t g);

  private:
    SimdAlignmentEngine(
        AlignmentType type,
        std::int8_t m,
        std::int8_t n,
        std::int8_t g);

    // void Realloc(
    //     std::uint64_t matrix_width,
    //     std::uint64_t matrix_height,
    //     std::uint8_t num_codes);

    // void Initialize(
    //     const char *sequence,
    //     const Graph &graph,
    //     std::uint64_t normal_matrix_width,
    //     std::uint64_t matrix_width,
    //     std::uint64_t matrix_height) noexcept;

    struct Implementation;
    std::unique_ptr<Implementation> pimpl_;
  };
} // namespace spoa

#endif // SIMD_ALIGNMENT_ENGINE_HPP_