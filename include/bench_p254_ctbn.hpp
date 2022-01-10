#pragma once
#include <CL/sycl.hpp>
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;

constexpr auto mod_p254 =
  21888242871839275222246405745257275088548364400416034343698204186575808495617_ZL;
constexpr auto mod_p254_bn = cbn::to_big_int(mod_p254);

using ff_p254_t = decltype(cbn::Zq(mod_p254));

sycl::event
benchmark_ff_p254_t_addition(sycl::queue& q,
                             uint32_t dim,
                             uint32_t wg_size,
                             uint32_t itr_count,
                             ff_p254_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p254_Addition>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p254_t tmp(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += ff_p254_t(70368744177664ul * i);
        }
#else
        ff_p254_t op(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);
        ff_p254_t tmp(
          904625697166532776746648320380374280103671755200316906558262375061821325304_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += op;
        }
#endif

        // every work item writes back to global memory
        // just to ensure kernel is not too much optimized by
        // compiler such that desired execution is skipped and
        // benchmark results are wrong !
        *(mem + idx) = tmp;
      });
  });
}

sycl::event
benchmark_ff_p254_t_subtraction(sycl::queue& q,
                                uint32_t dim,
                                uint32_t wg_size,
                                uint32_t itr_count,
                                ff_p254_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p254_Subtraction>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p254_t tmp(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp -= ff_p254_t(70368744177664ul * i);
        }
#else
        ff_p254_t op(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);
        ff_p254_t tmp(
          904625697166532776746648320380374280103671755200316906558262375061821325304_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp -= op;
        }
#endif

        // every work item writes back to global memory
        // just to ensure kernel is not too much optimized by
        // compiler such that desired execution is skipped and
        // benchmark results are wrong !
        *(mem + idx) = tmp;
      });
  });
}

sycl::event
benchmark_ff_p254_t_multiplication(sycl::queue& q,
                                   uint32_t dim,
                                   uint32_t wg_size,
                                   uint32_t itr_count,
                                   ff_p254_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p254_Multiplication>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p254_t tmp(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp *= ff_p254_t(70368744177664ul * i);
        }
#else
        ff_p254_t op(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);
        ff_p254_t tmp(
          904625697166532776746648320380374280103671755200316906558262375061821325304_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp *= op;
        }
#endif

        // every work item writes back to global memory
        // just to ensure kernel is not too much optimized by
        // compiler such that desired execution is skipped and
        // benchmark results are wrong !
        *(mem + idx) = tmp;
      });
  });
}

sycl::event
benchmark_ff_p254_t_division(sycl::queue& q,
                             uint32_t dim,
                             uint32_t wg_size,
                             uint32_t itr_count,
                             ff_p254_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p254_Division>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p254_t tmp(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp /= ff_p254_t(70368744177664ul * i);
        }
#else
        ff_p254_t op(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);
        ff_p254_t tmp(
          904625697166532776746648320380374280103671755200316906558262375061821325304_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp /= op;
        }
#endif

        // every work item writes back to global memory
        // just to ensure kernel is not too much optimized by
        // compiler such that desired execution is skipped and
        // benchmark results are wrong !
        *(mem + idx) = tmp;
      });
  });
}

sycl::event
benchmark_ff_p254_t_inversion(sycl::queue& q,
                              uint32_t dim,
                              uint32_t wg_size,
                              uint32_t itr_count,
                              ff_p254_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p254_Inversion>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p254_t tmp(0_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += static_cast<ff_p254_t>(
            cbn::mod_inv(ff_p254_t(70368744177664ul * i).data, mod_p254_bn));
        }
#else
        ff_p254_t tmp(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp = static_cast<ff_p254_t>(cbn::mod_inv(tmp.data, mod_p254_bn));
        }
#endif

        // every work item writes back to global memory
        // just to ensure kernel is not too much optimized by
        // compiler such that desired execution is skipped and
        // benchmark results are wrong !
        *(mem + idx) = tmp;
      });
  });
}

sycl::event
benchmark_ff_p254_t_exponentiation(sycl::queue& q,
                                   uint32_t dim,
                                   uint32_t wg_size,
                                   uint32_t itr_count,
                                   ff_p254_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p254_Exponentiation>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p254_t tmp(0_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += static_cast<ff_p254_t>(cbn::mod_exp(
            ff_p254_t(
              3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL)
              .data,
            ff_p254_t(70368744177664ul * (i + 1)).data,
            mod_p254_bn));
        }
#else
        ff_p254_t op(
          3618502788666131106986593281521497120414687020801267626233049500247285301247_ZL);
        ff_p254_t tmp(70368744177664_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp = static_cast<ff_p254_t>(
            cbn::mod_exp(op.data, tmp.data, mod_p254_bn));
        }
#endif

        // every work item writes back to global memory
        // just to ensure kernel is not too much optimized by
        // compiler such that desired execution is skipped and
        // benchmark results are wrong !
        *(mem + idx) = tmp;
      });
  });
}
