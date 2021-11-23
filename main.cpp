#include <bench_ctbn.hpp>
#include <types.hpp>
#include <utils.hpp>

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
  std::cout << "running on " << d.get_info<sycl::info::device::name>() << "\n"
            << std::endl;

  std::cout << "Addition on F(2^64 - 2^32 + 1)\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg"
            << "\t\t" << std::setw(24) << "ops/ sec" << std::endl;

  for (uint64_t i = 7; i <= 13; i++) {
    uint64_t dim = 1ul << i;
    tp start = std::chrono::system_clock::now();
    benchmark_ff_p_t_addition(q, dim, WG_SIZE, ITR_COUNT).wait();
    tp end = std::chrono::system_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nSubtraction on F(2^64 - 2^32 + 1)" << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg"
            << "\t\t" << std::setw(24) << "ops/ sec" << std::endl;

  for (uint64_t i = 7; i <= 13; i++) {
    uint64_t dim = 1ul << i;
    tp start = std::chrono::system_clock::now();
    benchmark_ff_p_t_subtraction(q, dim, WG_SIZE, ITR_COUNT).wait();
    tp end = std::chrono::system_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nMultiplication on F(2^64 - 2^32 + 1)" << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg"
            << "\t\t" << std::setw(24) << "ops/ sec" << std::endl;

  for (uint64_t i = 7; i <= 13; i++) {
    uint64_t dim = 1ul << i;
    tp start = std::chrono::system_clock::now();
    benchmark_ff_p_t_multiplication(q, dim, WG_SIZE, ITR_COUNT).wait();
    tp end = std::chrono::system_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  return 0;
}
