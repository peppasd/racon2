#include <fstream>

#include "bioparser/fastq_parser.hpp"
#include "biosoup/sequence.hpp"
#ifdef SPOA_USE_CEREAL
#include "cereal/archives/binary.hpp"
#endif
#include "gtest/gtest.h"

#include "spoa/spoa.hpp"

std::atomic<std::uint32_t> biosoup::Sequence::num_objects{0};

namespace spoa {
namespace test {

class SpoaTest: public ::testing::Test {
 public:
  void Setup(
      AlignmentType type,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g,
      std::int8_t e,
      std::int8_t q,
      std::int8_t c,
      bool quality) {
    auto p = bioparser::Parser<biosoup::Sequence>::Create<bioparser::FastqParser>(TEST_DATA);  // NOLINT
    s = p->Parse(-1);
    EXPECT_EQ(55, s.size());

    ae = AlignmentEngine::Create(type, m, n, g, e, q, c);
    gr = Graph();
    iq = quality;
  }

  void Align() {
    std::size_t ms = 0;
    for (const auto& it : s) {
      ms = std::max(ms, it->data.size());
    }
    ae->Prealloc(ms, 4);

    for (const auto& it : s) {
      auto a = ae->Align(it->data, gr);
      if (iq) {
        gr.AddAlignment(a, it->data, it->quality);
      } else {
        gr.AddAlignment(a, it->data);
      }
    }
  }

  void Check(const std::string& c) {
    EXPECT_EQ(c, gr.GenerateConsensus());

    auto msa = gr.GenerateMultipleSequenceAlignment();
    EXPECT_EQ(s.size(), msa.size());

    std::size_t rs = msa.front().size();
    std::vector<std::uint32_t> gc(rs, 0);
    for (const auto& it : msa) {
      EXPECT_EQ(rs, it.size());
      for (std::size_t i = 0; i < rs; ++i) {
        gc[i] += it[i] == '-' ? 1 : 0;
      }
    }
    for (const auto& it : gc) {
      EXPECT_GT(msa.size(), it);
    }

    for (std::uint32_t i = 0; i < msa.size(); ++i) {
      msa[i].erase(std::remove(msa[i].begin(), msa[i].end(), '-'), msa[i].end());  // NOLINT
      EXPECT_EQ(msa[i], s[i]->data);
    }
  }

  std::vector<std::unique_ptr<biosoup::Sequence>> s;
  std::unique_ptr<AlignmentEngine> ae;
  Graph gr;
  bool iq;
};

TEST(SpoaAlignmentTest, TypeError) {
  try {
    auto ae = AlignmentEngine::Create(static_cast<AlignmentType>(4), 1, -1, -1);
  } catch(std::invalid_argument& exception) {
    EXPECT_STREQ(
        exception.what(),
        "[spoa::AlignmentEngine::Create] error: invalid alignment type!");
  }
}

TEST(SpoaAlignmentTest, EmptyInput) {
  auto ae = AlignmentEngine::Create(AlignmentType::kSW, 1, -1, -1);
  Graph g{};
  auto a = ae->Align("", g);
  EXPECT_TRUE(a.empty());
}

TEST(SpoaAlignmentTest, LargeInput) {
  auto ae = AlignmentEngine::Create(AlignmentType::kSW, 1, -1, -1);
  try {
    ae->Prealloc(-1, 1);
  } catch (std::invalid_argument& exception) {
    EXPECT_EQ(
        std::string(exception.what()).substr(11),
        "AlignmentEngine::Prealloc] error: too large sequence!");
  }
  try {
    ae->Prealloc((1ULL << 31) - 1, -1);
  } catch (std::invalid_argument& exception) {
    EXPECT_EQ(
        std::string(exception.what()).substr(11),
        "AlignmentEngine::Prealloc] error: insufficient memory!");
  }
}

TEST_F(SpoaTest, Clear) {
  Setup(AlignmentType::kSW, 5, -4, -8, -8, -8, -8, false);
  Align();
  auto c = gr.GenerateConsensus();

  gr.Clear();
  Align();
  EXPECT_EQ(c, gr.GenerateConsensus());
}

#ifdef SPOA_USE_CEREAL
TEST_F(SpoaTest, Archive) {
  Setup(AlignmentType::kNW, 2, -5, -2, -2, -2, -2, true);

  {
    std::ofstream os("spoa.test.cereal");
    cereal::BinaryOutputArchive archive(os);
    archive(gr);
  }

  auto c = gr.GenerateConsensus();
  gr = {};

  {
    std::ifstream is("spoa.test.cereal");
    cereal::BinaryInputArchive archive(is);
    archive(gr);
  }

  EXPECT_EQ(c, gr.GenerateConsensus());
}
#endif

TEST_F(SpoaTest, Local) {
  Setup(AlignmentType::kSW, 5, -4, -8, -8, -8, -8, false);
  Align();

  std::string c =
      "AATGATGCGCTTTGTTGGCGCGGTGGCTTGATGCAGGGGCTAATCGACCTCTGGCAACCACTTTTCCATGAC"
      "AGGAGTTGAATATGGCATTCAGTAATCCCTTCGATGATCCGCAGGGAGCGTTTTACATATTGCGCAATGCGC"
      "AGGGGCAATTCAGTCTGTGGCCGCAACAATGCGTCTTACCGGCAGGCTGGGACATTGTGTGTCAGCCGCAGT"
      "CACAGGCGTCCTGCCAGCAGTGGCTGGAAGCCCACTGGCGTACTCTGACACCGACGAATTTTACCCAGTTGC"
      "AGGAGGCACAATGAGCCAGCATTTACCTTTGGTCGCCGCACAGCCCGGCATCTGGATGGCAGAAAAACTGTC"
      "AGAATTACCCTCCGCCTGGAGCGTGGCGCATTACGTTGAGTTAACCGGAGAGGTTGATTCGCCATTACTGGC"
      "CCGCGCGGTGGTTGCCGGACTAGCGCAAGCAGATACGCTTTACACGCGCAACCAAGGATTTCGG";

  Check(c);
}