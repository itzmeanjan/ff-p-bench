## 64-bit Prime Field Arithmetic Benchmark on GPU with CUDA Backend

```bash
DEVICE=gpu ON_THE_FLY=1 make cuda
```

```bash
Benchmark running on Tesla V100-SXM2-16GB

Addition on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		         147461 ns		     0.00878936 ns		           1.13774e+11
256  x  256		    1024		         386047 ns		     0.00575255 ns		           1.73836e+11
512  x  512		    1024		        1460205 ns		     0.00543969 ns		           1.83834e+11
1024 x 1024		    1024		        5868591 ns		     0.00546555 ns		           1.82964e+11

Subtraction on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		         115723 ns		     0.00689763 ns		           1.44977e+11
256  x  256		    1024		         323609 ns		     0.00482215 ns		           2.07376e+11
512  x  512		    1024		        1223632 ns		     0.00455838 ns		           2.19376e+11
1024 x 1024		    1024		        4726806 ns		     0.00440218 ns		            2.2716e+11

Multiplication on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		         323609 ns		      0.0192886 ns		           5.18441e+10
256  x  256		    1024		         857117 ns		       0.012772 ns		            7.8296e+10
512  x  512		    1024		        3254273 ns		      0.0121231 ns		           8.24871e+10
1024 x 1024		    1024		       12155884 ns		       0.011321 ns		            8.8331e+10

Division on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       28025878 ns		        1.67047 ns		           5.98633e+08
256  x  256		    1024		       61753357 ns		       0.920197 ns		           1.08672e+09
512  x  512		    1024		      247896057 ns		       0.923485 ns		           1.08285e+09
1024 x 1024		    1024		      784474121 ns		       0.730598 ns		           1.36874e+09

Inversion on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       22547486 ns		        1.34393 ns		           7.44084e+08
256  x  256		    1024		       49695801 ns		       0.740525 ns		           1.35039e+09
512  x  512		    1024		      199985107 ns		       0.745003 ns		           1.34228e+09
1024 x 1024		    1024		      698067871 ns		       0.650126 ns		           1.53816e+09

Exponentiation on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		        3399658 ns		       0.202635 ns		           4.93497e+09
256  x  256		    1024		       10707031 ns		       0.159547 ns		           6.26774e+09
512  x  512		    1024		       43558838 ns		       0.162269 ns		           6.16259e+09
1024 x 1024		    1024		      168333252 ns		       0.156773 ns		           6.37867e+09
```
