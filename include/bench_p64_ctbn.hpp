#pragma once
#include <CL/sycl.hpp>
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;

constexpr auto mod_p64 = 18446744069414584321_ZL;
constexpr auto mod_p64_bn = cbn::to_big_int(mod_p64);

using ff_p64_t = decltype(cbn::Zq(mod_p64));

sycl::event
benchmark_ff_p64_t_addition(sycl::queue& q,
                            uint32_t dim,
                            uint32_t wg_size,
                            uint32_t itr_count,
                            ff_p64_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p64_Addition>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p64_t tmp(2147483648_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += ff_p64_t(70368744177664ul * i);
        }
#else
        ff_p64_t op(2147483648_ZL);
        ff_p64_t tmp(576460752303423488_ZL);

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
benchmark_ff_p64_t_subtraction(sycl::queue& q,
                               uint32_t dim,
                               uint32_t wg_size,
                               uint32_t itr_count,
                               ff_p64_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p64_Subtraction>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p64_t tmp(2147483648_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp -= ff_p64_t(70368744177664ul * i);
        }
#else
        ff_p64_t op(2147483648_ZL);
        ff_p64_t tmp(576460752303423488_ZL);

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
benchmark_ff_p64_t_multiplication(sycl::queue& q,
                                  uint32_t dim,
                                  uint32_t wg_size,
                                  uint32_t itr_count,
                                  ff_p64_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p64_Multiplication>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p64_t tmp(2147483648_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp *= ff_p64_t(70368744177664ul * i);
        }
#else
        ff_p64_t op(2147483648_ZL);
        ff_p64_t tmp(576460752303423488_ZL);

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
benchmark_ff_p64_t_division(sycl::queue& q,
                            uint32_t dim,
                            uint32_t wg_size,
                            uint32_t itr_count,
                            ff_p64_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p64_Division>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p64_t tmp(2147483648_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp /= ff_p64_t(70368744177664ul * i);
        }
#else
        ff_p64_t op(2147483648_ZL);
        ff_p64_t tmp(576460752303423488_ZL);

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
benchmark_ff_p64_t_inversion(sycl::queue& q,
                             uint32_t dim,
                             uint32_t wg_size,
                             uint32_t itr_count,
                             ff_p64_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p64_Inversion>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p64_t tmp(0_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += static_cast<ff_p64_t>(
            cbn::mod_inv(ff_p64_t(70368744177664ul * i).data, mod_p64_bn));
        }
#else
        ff_p64_t tmp(576460752303423488_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp = static_cast<ff_p64_t>(cbn::mod_inv(tmp.data, mod_p64_bn));
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
benchmark_ff_p64_t_exponentiation(sycl::queue& q,
                                  uint32_t dim,
                                  uint32_t wg_size,
                                  uint32_t itr_count,
                                  ff_p64_t* const mem)
{
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelFF_p64_Exponentiation>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

#if ON_THE_FLY != 0
        ff_p64_t tmp(0_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += static_cast<ff_p64_t>(
            cbn::mod_exp(ff_p64_t(2147483648_ZL).data,
                         ff_p64_t(70368744177664ul * (i + 1)).data,
                         mod_p64_bn));
        }
#else
        ff_p64_t op(2147483648_ZL);
        ff_p64_t tmp(70368744177664_ZL);

        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp =
            static_cast<ff_p64_t>(cbn::mod_exp(op.data, tmp.data, mod_p64_bn));
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
