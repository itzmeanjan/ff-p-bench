#include <bench_ctbn.hpp>
#include <iomanip>
#include <iostream>
#include <types.hpp>

constexpr uint64_t ITR_COUNT = 1ul << 10;
constexpr uint64_t WG_SIZE = 1ul << 7;

int main(int argc, char **argv) {
#if defined CPU
  sycl::device d{sycl::cpu_selector{}};
#elif defined GPU
  sycl::device d{sycl::gpu_selector{}};
#elif defined HOST
  sycl::device d{sycl::host_selector{}};
#else
  sycl::device d{sycl::default_selector{}};
#endif

  sycl::queue q{d};

  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint64_t i = 7; i <= 13; i++) {
    uint64_t dim = 1ul << i;
    tp start = std::chrono::system_clock::now();
    benchmark_ff_p_t_addition(q, dim, WG_SIZE, ITR_COUNT).wait();
    tp end = std::chrono::system_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right
              << ITR_COUNT << "\t\t" << std::setw(15) << std::right << tm
              << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * ITR_COUNT) << " ns"
              << std::endl;
  }

  return 0;
}
