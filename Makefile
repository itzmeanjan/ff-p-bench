CXX = dpcpp
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
INCLUDES = -I./include
PROG = run
DFLAGS = -D$(shell echo $(or $(DEVICE),default) | tr a-z A-Z)

$(PROG): main.o utils.o bench_p64_ctbn.o bench_p256_ctbn.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

main.o: main.cpp include/bench_p64_ctbn.hpp include/bench_p256_ctbn.hpp include/types.hpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_p64_ctbn.o: bench_p64_ctbn.cpp include/bench_p64_ctbn.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_p256_ctbn.o: bench_p256_ctbn.cpp include/bench_p256_ctbn.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' -o -name 'test' -o  -name '__pycache__' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
