## 254-bit Prime Field Arithmetic Benchmark on GPU with CUDA Backend

```bash
DEVICE=gpu ON_THE_FLY=0 make cuda
```

```bash
Benchmark running on Tesla V100-SXM2-16GB

Addition on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		         196533 ns		      0.0117143 ns		           8.53659e+10
256  x  256		    1024		         599853 ns		     0.00893851 ns		           1.11876e+11
512  x  512		    1024		        2343994 ns		     0.00873206 ns		           1.14521e+11
1024 x 1024		    1024		        8756348 ns		     0.00815498 ns		           1.22624e+11

Subtraction on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		         202637 ns		      0.0120781 ns		           8.27944e+10
256  x  256		    1024		         673828 ns		      0.0100408 ns		           9.95935e+10
512  x  512		    1024		        2467042 ns		     0.00919045 ns		           1.08809e+11
1024 x 1024		    1024		        9998535 ns		     0.00931186 ns		            1.0739e+11

Multiplication on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       31846436 ns		         1.8982 ns		           5.26816e+08
256  x  256		    1024		      151574463 ns		        2.25864 ns		           4.42745e+08
512  x  512		    1024		      739296142 ns		        2.75409 ns		           3.63096e+08
1024 x 1024		    1024		     2951131104 ns		        2.74846 ns		           3.63841e+08

Division on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		      197851074 ns		        11.7928 ns		           8.47972e+07
256  x  256		    1024		      712227051 ns		         10.613 ns		            9.4224e+07
512  x  512		    1024		     1978044434 ns		        7.36879 ns		           1.35707e+08
1024 x 1024		    1024		     7296841797 ns		        6.79571 ns		           1.47152e+08

Inversion on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		      177291015 ns		        10.5674 ns		           9.46309e+07
256  x  256		    1024		      432279297 ns		        6.44146 ns		           1.55244e+08
512  x  512		    1024		     1726179687 ns		        6.43052 ns		           1.55508e+08
1024 x 1024		    1024		     6429169922 ns		        5.98763 ns		           1.67011e+08

Exponentiation on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		      715103516 ns		        42.6235 ns		           2.34612e+07
256  x  256		    1024		     2686921875 ns		        40.0383 ns		           2.49761e+07
512  x  512		    1024		     8996812500 ns		        33.5157 ns		           2.98367e+07
1024 x 1024		    1024		    34479550781 ns		        32.1116 ns		           3.11414e+07
```
