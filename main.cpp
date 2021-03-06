#include <bench_p254_ctbn.hpp>
#include <bench_p64_ctbn.hpp>
#include <utils.hpp>

#if ON_THE_FLY == 0
#pragma message(                                                               \
  "Compiling kernels such that pre-computed big integer operands are used inside benchmark kernel !")
#else
#pragma message(                                                               \
  "Compiling kernels such that big integer operands are all computed on-the-fly !")
#endif

constexpr size_t ITR_COUNT = 1ul << 10;
constexpr size_t WG_SIZE = 1ul << 6;

int
main(int argc, char** argv)
{
#if defined CPU
  sycl::device d{ sycl::cpu_selector{} };
#elif defined GPU
  sycl::device d{ sycl::gpu_selector{} };
#elif defined HOST
  sycl::device d{ sycl::host_selector{} };
#else
  sycl::device d{ sycl::default_selector{} };
#endif

  // enabling queue profiling required for SYCL event based kernel execution
  // timing
  sycl::queue q{ d, sycl::property::queue::enable_profiling{} };
  std::cout << "Benchmark running on " << d.get_info<sycl::info::device::name>()
            << "\n"
            << std::endl;

  // here starts 64-bit prime field benchmarks

  std::cout << "Addition on F(2^64 - 2^32 + 1)\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p64_t* mem = static_cast<ff_p64_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p64_t), q));

    sycl::event evt =
      benchmark_ff_p64_t_addition(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nSubtraction on F(2^64 - 2^32 + 1)" << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p64_t* mem = static_cast<ff_p64_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p64_t), q));

    sycl::event evt =
      benchmark_ff_p64_t_subtraction(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nMultiplication on F(2^64 - 2^32 + 1)" << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p64_t* mem = static_cast<ff_p64_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p64_t), q));

    sycl::event evt =
      benchmark_ff_p64_t_multiplication(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nDivision on F(2^64 - 2^32 + 1)" << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p64_t* mem = static_cast<ff_p64_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p64_t), q));

    sycl::event evt =
      benchmark_ff_p64_t_division(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nInversion on F(2^64 - 2^32 + 1)" << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p64_t* mem = static_cast<ff_p64_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p64_t), q));

    sycl::event evt =
      benchmark_ff_p64_t_inversion(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nExponentiation on F(2^64 - 2^32 + 1)" << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p64_t* mem = static_cast<ff_p64_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p64_t), q));

    sycl::event evt =
      benchmark_ff_p64_t_exponentiation(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  // now starts 254-bit prime field benchmarks

  std::cout << "\nAddition on "
               "F("
               "218882428718392752222464057452572750885483644004160343436982041"
               "86575808495617)\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p254_t* mem = static_cast<ff_p254_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p254_t), q));

    sycl::event evt =
      benchmark_ff_p254_t_addition(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nSubtraction on "
               "F("
               "218882428718392752222464057452572750885483644004160343436982041"
               "86575808495617)"
            << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p254_t* mem = static_cast<ff_p254_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p254_t), q));

    sycl::event evt =
      benchmark_ff_p254_t_subtraction(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nMultiplication on "
               "F("
               "218882428718392752222464057452572750885483644004160343436982041"
               "86575808495617)"
            << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p254_t* mem = static_cast<ff_p254_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p254_t), q));

    sycl::event evt =
      benchmark_ff_p254_t_multiplication(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nDivision on "
               "F("
               "218882428718392752222464057452572750885483644004160343436982041"
               "86575808495617)"
            << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p254_t* mem = static_cast<ff_p254_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p254_t), q));

    sycl::event evt =
      benchmark_ff_p254_t_division(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nInversion on "
               "F("
               "218882428718392752222464057452572750885483644004160343436982041"
               "86575808495617)"
            << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p254_t* mem = static_cast<ff_p254_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p254_t), q));

    sycl::event evt =
      benchmark_ff_p254_t_inversion(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  std::cout << "\nExponentiation on "
               "F("
               "218882428718392752222464057452572750885483644004160343436982041"
               "86575808495617)"
            << std::endl;
  std::cout << "\n"
            << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(24) << "per op"
            << "\t\t" << std::setw(20) << "ops/ sec" << std::endl;

  for (size_t i = 7; i <= 10; i++) {
    size_t dim = 1ul << i;
    ff_p254_t* mem = static_cast<ff_p254_t*>(
      sycl::malloc_device(dim * dim * sizeof(ff_p254_t), q));

    sycl::event evt =
      benchmark_ff_p254_t_exponentiation(q, dim, WG_SIZE, ITR_COUNT, mem);
    evt.wait();
    sycl::free(mem, q);

    sycl::cl_ulong tm = time_event(evt);
    double tm_per_op = (double)tm / (double)(dim * dim * ITR_COUNT);
    print_benchmark_table_row(dim, ITR_COUNT, tm, tm_per_op);
  }

  return 0;
}
