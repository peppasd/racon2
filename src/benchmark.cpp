#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#include <spoa/graph.hpp>
#include <spoa/alignment_engine.hpp>

int main(int argc, char **argv)
{

  std::vector<std::vector<std::string>> windows;
  std::ifstream file("test/data/window.txt");
  std::string line;
  std::int32_t window_count = -1;

  while (getline(file, line))
  {
    if ((line[0] - '0') > 0 && (line[0] - '9') < 0)
    {
      windows.emplace_back(std::vector<std::string>());
      ++window_count;
    }
    else
    {
      windows[window_count].push_back(line);
    }
  }

  auto alignment_engine = spoa::AlignmentEngine::Create(
      spoa::AlignmentType::Simd, 3, -5, -3);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < window_count; i++)
  {

    spoa::Graph graph{};

    for (const auto &it : windows[i])
    {
      auto alignment = alignment_engine->Align(it, graph);
      graph.AddAlignment(alignment, it);
    }

    auto consensus = graph.GenerateConsensus();

    std::cerr << std::endl
              << i + 1 << "/66 >Consensus LN:i:" << consensus.size() << std::endl
              << consensus << std::endl;
  }

  auto duration = std::chrono::steady_clock::now() - start;
  int serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  std::cout << "benchmark done" << std::endl;
  std::cout << "time: " << serial_time << " ms" << std::endl;

  return 0;
}