// Copyright (c) 2020 Robert Vaser

#ifndef SISD_ALIGNMENT_ENGINE_HPP_
#define SISD_ALIGNMENT_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "spoa/alignment_engine.hpp"
#include "spoa/graph.hpp"

namespace spoa {

class SisdAlignmentEngine: public AlignmentEngine {
 public:
  SisdAlignmentEngine(const SisdAlignmentEngine&) = delete;
  SisdAlignmentEngine& operator=(const SisdAlignmentEngine&) = delete;

  SisdAlignmentEngine(SisdAlignmentEngine&&) = default;
  SisdAlignmentEngine& operator=(SisdAlignmentEngine&&) = default;

  ~SisdAlignmentEngine() = default;


  static std::unique_ptr<AlignmentEngine> Create(
      AlignmentType type,
      AlignmentSubtype subtype,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g,
      std::int8_t e,
      std::int8_t q,
      std::int8_t c);

  void Prealloc(
      std::uint32_t max_sequence_len,
      std::uint8_t alphabet_size) override;

  Alignment Align(
      const char* sequence, std::uint32_t sequence_len,
      const Graph& graph,
      std::int32_t* score) override;

 private:
  SisdAlignmentEngine(
      AlignmentType type,
      AlignmentSubtype subtype,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g,
      std::int8_t e,
      std::int8_t q,
      std::int8_t c);

  Alignment Linear(
      std::uint32_t sequence_len,
      const Graph& graph,
      std::int32_t* score) noexcept;

  Alignment Affine(
      std::uint32_t sequence_len,
      const Graph& graph,
      std::int32_t* score) noexcept;

  Alignment Convex(
      std::uint32_t sequence_len,
      const Graph& graph,
      std::int32_t* score) noexcept;

  void Realloc(
      std::uint64_t matrix_width,
      std::uint64_t matrix_height,
      std::uint8_t num_codes);

  void Initialize(
      const char* sequence, std::uint32_t sequence_len,
      const Graph& graph) noexcept;

    //void parallel_aligment(auto it, int);
    // void parallel_aligment(
    //   std::vector<spoa::Graph::Node *>*a,
    //   std::uint32_t sequence_len,
    //   std::int32_t* score,
    //   int tid);

    void parallel_aligment(
      spoa::Graph::Node* it,
      int tid);
    

  struct Implementation;
  struct threadclass;
  std::unique_ptr<Implementation> pimpl_;
  std::unique_ptr<threadclass> workers;
  std::atomic<int> complete_thread{0};
  std::atomic<int> debug{0};

  std::int32_t max_score;
  std::uint64_t matrix_width;
  std::uint32_t max_i;
  std::uint32_t max_j;
  bool* next_round;
};

}  // namespace spoa

#endif  // SISD_ALIGNMENT_ENGINE_HPP_
