CXX = dpcpp
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
INCLUDES = -I./include
PROG = run
DFLAGS = -D$(shell echo $(or $(DEVICE),default) | tr a-z A-Z)

$(PROG): main.o bench_ctbn.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

main.o: main.cpp include/bench_ctbn.hpp include/types.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_ctbn.o: bench_ctbn.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' -o -name 'test' -o  -name '__pycache__' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
