#include <bench_ctbn.hpp>

sycl::event benchmark_ff_p_addition(sycl::queue &q, const uint32_t dim,
                                    const uint32_t wg_size,
                                    const uint32_t itr_count) {
  sycl::event evt = q.parallel_for(
      sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
      [=](sycl::nd_item<2> it) {
        const uint64_t r = it.get_global_id(0);
        const uint64_t c = it.get_global_id(1);

        ff_p_t op1{1ul << 31};
        ff_p_t op2{1ul << 59};

        ff_p_t tmp{0ul};
        for (uint64_t i = 0ul; i < itr_count; i++) {
          tmp += (op1 + op2);
        }
      });
  return evt;
}
