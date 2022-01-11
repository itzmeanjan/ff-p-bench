#pragma once
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

// Prints a row of benchmark table; as shown
// https://github.com/itzmeanjan/ff_p_bench/blob/61838f7/benchmarks/254-bit-on-gpu.md#L32
void
print_benchmark_table_row(size_t dim,
                          size_t itr_cnt,
                          sycl::cl_ulong total_tm,
                          double tm_per_op)
{
  std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
            << std::right << dim << "\t\t" << std::setw(8) << std::right
            << itr_cnt << "\t\t" << std::setw(15) << std::right << total_tm
            << " ns"
            << "\t\t" << std::setw(15) << std::right << tm_per_op << " ns"
            << "\t\t" << std::setw(22) << std::right << 1e9 / tm_per_op
            << std::endl;
}

// Given a SYCL event ( obtained as result of submitting kernel ) computes
// actual execution time of job with nanosecond level granularity
//
// Make sure SYCL queue has profiling enabled, otherwise it'll end up panicing !
sycl::cl_ulong
time_event(sycl::event evt)
{
  sycl::cl_ulong start =
    evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  sycl::cl_ulong end =
    evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  return end - start;
}
