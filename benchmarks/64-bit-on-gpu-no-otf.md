## 64-bit Prime Field Arithmetic Benchmark on GPU with CUDA Backend

```bash
DEVICE=gpu ON_THE_FLY=0 make cuda
```

```bash
Benchmark running on Tesla V100-SXM2-16GB

Addition on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		          76799 ns		     0.00457758 ns		           2.18456e+11
256  x  256		    1024		         218112 ns		     0.00325012 ns		           3.07681e+11
512  x  512		    1024		         772095 ns		     0.00287628 ns		           3.47672e+11
1024 x 1024		    1024		        3023872 ns		      0.0028162 ns		           3.55088e+11

Subtraction on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		          53248 ns		     0.00317383 ns		           3.15077e+11
256  x  256		    1024		         149504 ns		     0.00222778 ns		           4.48877e+11
512  x  512		    1024		         528384 ns		     0.00196838 ns		           5.08031e+11
1024 x 1024		    1024		        2063359 ns		     0.00192165 ns		           5.20385e+11

Multiplication on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		         209920 ns		      0.0125122 ns		            7.9922e+10
256  x  256		    1024		         552961 ns		     0.00823976 ns		           1.21363e+11
512  x  512		    1024		        2121728 ns		     0.00790405 ns		           1.26517e+11
1024 x 1024		    1024		        7871488 ns		     0.00733089 ns		           1.36409e+11

Division on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       19107840 ns		        1.13892 ns		           8.78028e+08
256  x  256		    1024		       46345215 ns		       0.690598 ns		           1.44802e+09
512  x  512		    1024		      202934280 ns		       0.755989 ns		           1.32277e+09
1024 x 1024		    1024		      600725495 ns		       0.559469 ns		           1.78741e+09

Inversion on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       19352600 ns		         1.1535 ns		           8.66923e+08
256  x  256		    1024		       41293823 ns		       0.615326 ns		           1.62516e+09
512  x  512		    1024		      164434875 ns		       0.612568 ns		           1.63247e+09
1024 x 1024		    1024		      594865112 ns		       0.554011 ns		           1.80502e+09

Exponentiation on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       13501343 ns		       0.804743 ns		           1.24263e+09
256  x  256		    1024		       35887085 ns		       0.534759 ns		              1.87e+09
512  x  512		    1024		      138602417 ns		       0.516334 ns		           1.93673e+09
1024 x 1024		    1024		      536241211 ns		       0.499414 ns		           2.00235e+09
```
