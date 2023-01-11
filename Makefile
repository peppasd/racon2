CXXCOMPILER = clang++
CXXFLAGS = -std=c++20 -Iinclude/ -DSIMDPP_DISPATCH_ARCH5=SIMDPP_ARCH_X86_AVX2 -mavx2 -O2

racon: src/main.cpp src/graph.cpp src/alignment_engine.cpp src/simd_alignment_engine.cpp src/sisd_alignment_engine.cpp
	$(CXXCOMPILER) $(CXXFLAGS) $^ -o racon

benchmark: src/benchmark.cpp src/graph.cpp src/alignment_engine.cpp src/simd_alignment_engine.cpp src/sisd_alignment_engine.cpp
	$(CXXCOMPILER) $(CXXFLAGS) $^ -o benchmark

test: src/test.cpp
	$(CXXCOMPILER) src/test.cpp $(CXXFLAGS)  -o test
