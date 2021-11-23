#pragma once
#include <CL/sycl.hpp>
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;
using ff_p_t = decltype(cbn::Zq(18446744069414584321_ZL));

sycl::event benchmark_ff_p_t_addition(sycl::queue &q, const uint32_t dim,
                                      const uint32_t wg_size,
                                      const uint32_t itr_count);

sycl::event benchmark_ff_p_t_subtraction(sycl::queue &q, const uint32_t dim,
                                         const uint32_t wg_size,
                                         const uint32_t itr_count);

sycl::event benchmark_ff_p_t_multiplication(sycl::queue &q, const uint32_t dim,
                                            const uint32_t wg_size,
                                            const uint32_t itr_count);
