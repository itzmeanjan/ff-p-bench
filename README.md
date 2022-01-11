# ff_p_bench

Benchmarking Finite Field Arithmetics on Accelerators, using SYCL

## Introduction

I've been exploring finite field arithmetics libraries which I can use for offloading computation to accelerator devices i.e. CPU, GPU. Most of the arbitrary precision modular arithmetic libraries I came across, makes use of dynamic memory allocation, which is prohibited when running field arithmetics inside kernels, which are offloaded to CPU/ GPU, using backends like OpenCL, CUDA. Recently I discovered `ctbignum`, written by [Niek J. Bouman](https://github.com/niekbouman/ctbignum), which is wholely based on modern C++ features i.e. `std::array`, `std::integer_sequence` etc., fully avoiding any need of dynamic allocation. All field elements ( read limbs if field element can't be stored in single machine word ) are kept in stack/ registers.

With this setup, I write benchmark code for two prime fields, while offloading parallel computation to CPU/ GPU. I'm using SYCL as frontend framework, while OpenCL/ CUDA/ HIP can be used as backend, given that SYCL implementation can produce code for such backend.

These are two prime fields, I'm benchmarking on. These two prime fields are particularly of my interest, which is why I choose them.

- F(18446744069414584321) --- **64-bit Prime Field**
- F(21888242871839275222246405745257275088548364400416034343698204186575808495617) --- **254-bit Prime Field**

> Note: For compiling `ctbignum` down to GPU digestable **SPIRV-64** code, I had to make small modification in `ctbignum`. If interested, read [more](https://github.com/niekbouman/ctbignum/pull/48).

## Setup

- I'm on

```bash
$ lsb_release -d

Description:    Ubuntu 20.04.3 LTS
```

- I've compiled Intel DPC++ compiler from source, with CUDA support, while following guide [here](https://intel.github.io/llvm-docs/GetStartedGuide.html#prerequisites)

- Compiler version is

```bash
$ clang++ --version

clang version 14.0.0 (https://github.com/intel/llvm c690ac8d771e8bb1a1be651872b782f4044d936c)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/ubuntu/sycl_workspace/llvm/build/bin
```

- I've `libstdc++-10` installed so that I can use C++20 features

```bash
sudo apt-get install libstdc++-10-dev
```

- Clone `ctbignum` & include headers in include path

```bash
cd ~
git clone https://github.com/itzmeanjan/ctbignum.git

pushd ctbignum
git checkout e8fdb0d6f7d304fb1eed2029078d3f653c4f67db # **important**
sudo cp -r include/ctbignum /usr/local/include
popd
```

- Also you'll need `make` and standard system development tools.

## Usage

- Run benchmark

```bash
# either

# Prime field elements are pre-calculated & 
# those operands are used inside kernel body ( i.e. loop )
ON_THE_FLY=0 make


# or

# Prime field elements are computed on-the-fly ( runtime )
# and what they are is dictated by work-item index, iteration 
# round index etc.
ON_THE_FLY=1 make
```

- Clean up

```bash
make clean
```

- If you make any changes in soource, reformat

```bash
# given that you've clang-format installed
make format
```

- It's possible to target some specific accelerator device, if you have multiple

```bash
DEVICE=cpu      make          # target first CPU available
DEVICE=gpu      make          # target first GPU available
DEVICE=host     make          # target host, *always* available
DEVICE=default  make          # no need to mention device here, it's default case
```

> All above kernels are JIT compiled, for AOT try taking a look [here](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html).

- I've added build recipe for AOT compiling all benchmark kernels, targeting CPU, using `avx*`/ `sse4.2` instructions

```bash
DEVICE=cpu ON_THE_FLY={0,1} make aot_cpu
```

- If you've access to Nvidia GPU, you can compile kernels using experimental CUDA backend of DPC++

```bash
DEVICE=gpu ON_THE_FLY={0,1} make cuda
```

**I suggest checking Makefile living in root of repository.**

## Benchmark

For running benchmark in data parallel environment, I make use of 2D compute grid, where each cell of square matrix of dimension *N x N*, is one work-item. Each work-item computes some specific field arithmetic operation on given operands, M-many times. So in each round of benchmarking, when some specific arithmetic operator is chosen, for prime field **F_p**; that operation is concurrently run on chosen accelerator device `N x N x M` times. These benchmarks don't include any (host-to-device and vice-versa) data transfer. As suggested/ corrected by @niekbauman, I've modified kernels to make use of global memory and ensure that each work-item writes some accumulated data back to designated location in global memory at end of work-item's compute cycle, so that compiler doesn't end up optimizing too much that kernel actually doesn't do its desired job and I collect wrong metrics.

### On GPU/ CUDA

`ON_THE_FLY` | Prime Field | Results
--- | --- | ---
0 | 64 -bit | [benchmarks/64-bit-on-gpu-no-otf.md](benchmarks/64-bit-on-gpu-no-otf.md)
1 | 64 -bit | [benchmarks/64-bit-on-gpu-otf.md](benchmarks/64-bit-on-gpu-otf.md)
0 | 254 -bit | [benchmarks/254-bit-on-gpu-no-otf.md](benchmarks/254-bit-on-gpu-no-otf.md)
1 | 254 -bit | [benchmarks/254-bit-on-gpu-otf.md](benchmarks/254-bit-on-gpu-otf.md)
