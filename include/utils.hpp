#pragma once
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

void print_benchmark_table_row(const uint64_t dim, const uint64_t itr_cnt,
                               const int64_t total_tm, const double tm_per_op);

// Given a SYCL event ( obtained as result of submitting kernel ) computes 
// actual execution time of job with nanosecond level granularity
//
// Make sure SYCL queue has profiling enabled, otherwise it'll end up panicing !
sycl::cl_ulong time_event(sycl::event evt) {
    sycl::cl_ulong start = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
    sycl::cl_ulong end = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
    return end - start;
}
