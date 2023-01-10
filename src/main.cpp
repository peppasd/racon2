#include <iostream>

#include <spoa/graph.hpp>
#include <spoa/alignment_engine.hpp>

int main(int argc, char **argv)
{

  std::vector<std::string> sequences = {
      "CATAAAAGAACGTAGGTCGCCCGTCCGTAACCTGTCGGATCACCGGAAAGGACCCGTAAAGTGATAATGAT",
      "ATAAAGGCAGTCGCTCTGTAAGCTGTCGATTCACCGGAAAGATGGCGTTACCACGTAAAGTGATAATGATTAT",
      "ATCAAAGAACGTGTAGCCTGTCCGTAATCTAGCGCATTTCACACGAGACCCGCGTAATGGG",
      "CGTAAATAGGTAATGATTATCATTACATATCACAACTAGGGCCGTATTAATCATGATATCATCA",
      "GTCGCTAGAGGCATCGTGAGTCGCTTCCGTACCGCAAGGATGACGAGTCACTTAAAGTGATAAT",
      "CCGTAACCTTCATCGGATCACCGGAAAGGACCCGTAAATAGACCTGATTATCATCTACAT"};

  auto alignment_engine = spoa::AlignmentEngine::Create(
      spoa::AlignmentType::Sisd, 3, -5, -3);

  spoa::Graph graph{};

  for (const auto &it : sequences)
  {
    auto alignment = alignment_engine->Align(it, graph);
    graph.AddAlignment(alignment, it);
  }

  auto consensus = graph.GenerateConsensus();

  std::cerr << std::endl
            << ">Consensus LN:i:" << consensus.size() << std::endl
            << consensus << std::endl;

  auto msa = graph.GenerateMultipleSequenceAlignment();

  for (const auto &it : msa)
  {
    std::cerr << it << std::endl;
  }

  std::cout << std::endl;

  return 0;
}