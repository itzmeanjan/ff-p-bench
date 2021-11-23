#pragma once
#include <CL/sycl.hpp>
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;

constexpr auto mod_p256 =
    21888242871839275222246405745257275088548364400416034343698204186575808495617_ZL;
constexpr auto mod_p256_bn = cbn::to_big_int(mod_p256);

using ff_p256_t = decltype(cbn::Zq(mod_p256));

sycl::event benchmark_ff_p256_t_addition(sycl::queue &q, const uint32_t dim,
                                         const uint32_t wg_size,
                                         const uint32_t itr_count);

sycl::event benchmark_ff_p256_t_subtraction(sycl::queue &q, const uint32_t dim,
                                            const uint32_t wg_size,
                                            const uint32_t itr_count);

sycl::event benchmark_ff_p256_t_multiplication(sycl::queue &q,
                                               const uint32_t dim,
                                               const uint32_t wg_size,
                                               const uint32_t itr_count);

sycl::event benchmark_ff_p256_t_division(sycl::queue &q, const uint32_t dim,
                                         const uint32_t wg_size,
                                         const uint32_t itr_count);

sycl::event benchmark_ff_p256_t_inversion(sycl::queue &q, const uint32_t dim,
                                          const uint32_t wg_size,
                                          const uint32_t itr_count);

sycl::event benchmark_ff_p256_t_exponentiation(sycl::queue &q,
                                               const uint32_t dim,
                                               const uint32_t wg_size,
                                               const uint32_t itr_count);
