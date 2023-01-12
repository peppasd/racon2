CXXFLAGS = -std=c++20 -I/usr/local/include

test4: test4.o /usr/local/lib/libspoa.a  #test_null.o
	g++ $^ -std=c++20 -o test4

test4.o: test.cpp
	g++ test.cpp $(CXXFLAGS) -c -o test4.o