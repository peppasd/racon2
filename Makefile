CXXCOMPILER = clang++
CXXFLAGS = -std=c++20 -Iinclude/

racon: main.o graph.o alignment_engine.o simd_alignment_engine.o
	$(CXXCOMPILER) $^ -o racon
	rm *.o

main.o: src/main.cpp
	$(CXXCOMPILER) src/main.cpp $(CXXFLAGS) -c -o main.o

graph.o: src/graph.cpp
	$(CXXCOMPILER) src/graph.cpp $(CXXFLAGS) -c -o graph.o

alignment_engine.o: src/alignment_engine.cpp
	$(CXXCOMPILER) src/alignment_engine.cpp $(CXXFLAGS) -c -o alignment_engine.o

simd_alignment_engine.o: src/simd_alignment_engine.cpp
	$(CXXCOMPILER) src/simd_alignment_engine.cpp $(CXXFLAGS) -DSIMDPP_DISPATCH_ARCH5=SIMDPP_ARCH_X86_AVX2 -mavx2 -c -o simd_alignment_engine.o
