#ifndef SISD_ALIGNMENT_ENGINE_HPP_
#define SISD_ALIGNMENT_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "spoa/alignment_engine.hpp"

namespace spoa
{

  std::unique_ptr<AlignmentEngine> CreateSisdAlignmentEngine(
      std::int8_t m,
      std::int8_t n,
      std::int8_t g);

  class SisdAlignmentEngine : public AlignmentEngine
  {
  public:
    SisdAlignmentEngine(const SisdAlignmentEngine &) = delete;
    SisdAlignmentEngine &operator=(const SisdAlignmentEngine &) = delete;

    SisdAlignmentEngine(SisdAlignmentEngine &&) = default;
    SisdAlignmentEngine &operator=(SisdAlignmentEngine &&) = delete;

    ~SisdAlignmentEngine() = default;

    static std::unique_ptr<AlignmentEngine> Create(
        AlignmentType type,
        std::int8_t m,
        std::int8_t n,
        std::int8_t g);

    Alignment Align(
        const char *sequence, std::uint32_t sequence_len,
        const Graph &graph,
        std::int32_t *score) override;

    friend std::unique_ptr<AlignmentEngine> CreateSisdAlignmentEngine(
        std::int8_t m,
        std::int8_t n,
        std::int8_t g);

  private:
    SisdAlignmentEngine(
        AlignmentType type,
        std::int8_t m,
        std::int8_t n,
        std::int8_t g);

    struct Implementation;
    std::unique_ptr<Implementation> pimpl_;
  };
} // namespace spoa

#endif // SISD_ALIGNMENT_ENGINE_HPP_