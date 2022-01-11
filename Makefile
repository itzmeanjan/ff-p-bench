CXX = dpcpp
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl-targets=nvptx64-nvidia-cuda
SYCLAOTFLAGS = -fsycl-default-sub-group-size 32
INCLUDES = -I./include
PROG = run
DEV_FLAGS = -D$(shell echo $(or $(DEVICE),default) | tr a-z A-Z)
ON_THE_FLY_FLAGS = -DON_THE_FLY=$(shell echo $(or $(ON_THE_FLY),0))

$(PROG): main.cpp include/bench_p64_ctbn.hpp include/bench_p254_ctbn.hpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DEV_FLAGS) $(ON_THE_FLY_FLAGS) $(INCLUDES) $< -o $@
	./$@

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' -o -name 'test' -o  -name '__pycache__' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla

aot_cpu:
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(DEV_FLAGS) $(ON_THE_FLY_FLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=avx512" main.cpp -o $(PROG); \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(DEV_FLAGS) $(ON_THE_FLY_FLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=avx2" main.cpp -o $(PROG); \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(DEV_FLAGS) $(ON_THE_FLY_FLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=avx" main.cpp -o $(PROG); \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(DEV_FLAGS) $(ON_THE_FLY_FLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64 -Xs "-march=sse4.2" main.cpp -o $(PROG); \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi
	./$(PROG)

cuda:
	# make sure you've built `clang++` with CUDA support
	# check https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda
	clang++ $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) $(DEV_FLAGS) $(ON_THE_FLY_FLAGS) $(INCLUDES) main.cpp -o $(PROG)
	./$(PROG)
