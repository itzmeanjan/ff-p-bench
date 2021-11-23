## 64-bit Prime Field Arithmetic Benchmark on CPU with OpenCL Backend

```bash
$ DEVICE=cpu make && ./run

Benchmark running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
```

### Modular Addition

```bash
Addition on F(2^64 - 2^32 + 1)


  dimension             iterations                        total                           per op                            ops/ sec                                                                                                                                            
128  x  128                 1024                      565484289 ns                      33.7055 ns                         2.96688e+07
256  x  256                 1024                         103320 ns                   0.00153959 ns                         6.49524e+11
512  x  512                 1024                          98176 ns                  0.000365734 ns                         2.73423e+12
1024 x 1024                 1024                         133678 ns                  0.000124497 ns                          8.0323e+12
2048 x 2048                 1024                         279722 ns                  6.51279e-05 ns                         1.53544e+13
4096 x 4096                 1024                         945265 ns                  5.50217e-05 ns                         1.81747e+13
8192 x 8192                 1024                        3615225 ns                  5.26084e-05 ns                         1.90084e+13
```

### Modular Subtraction

```bash
Subtraction on F(2^64 - 2^32 + 1)


  dimension             iterations                        total                           per op                            ops/ sec                                                                                                                                            
128  x  128                 1024                        5126204 ns                     0.305546 ns                         3.27283e+09
256  x  256                 1024                          74317 ns                   0.00110741 ns                         9.03008e+11
512  x  512                 1024                         106052 ns                  0.000395074 ns                         2.53117e+12
1024 x 1024                 1024                         144804 ns                  0.000134859 ns                         7.41514e+12
2048 x 2048                 1024                         283981 ns                  6.61195e-05 ns                         1.51241e+13
4096 x 4096                 1024                         930831 ns                  5.41815e-05 ns                         1.84565e+13
8192 x 8192                 1024                        3482146 ns                  5.06719e-05 ns                         1.97348e+13
```

### Modular Multiplication

```bash
Multiplication on F(2^64 - 2^32 + 1)


  dimension             iterations                        total                           per op                            ops/ sec                                                                                                                                            
128  x  128                 1024                        5457204 ns                     0.325275 ns                         3.07432e+09
256  x  256                 1024                          84352 ns                   0.00125694 ns                         7.95581e+11
512  x  512                 1024                          70456 ns                  0.000262469 ns                         3.80997e+12
1024 x 1024                 1024                         130183 ns                  0.000121242 ns                         8.24794e+12
2048 x 2048                 1024                         296479 ns                  6.90294e-05 ns                         1.44866e+13
4096 x 4096                 1024                         923944 ns                  5.37806e-05 ns                         1.85941e+13
8192 x 8192                 1024                        3306693 ns                  4.81187e-05 ns                         2.07819e+13
```

### Modular Division

```bash
Division on F(2^64 - 2^32 + 1)


  dimension             iterations                        total                           per op                            ops/ sec                                                                                                                                            
128  x  128                 1024                        5220513 ns                     0.311167 ns                         3.21371e+09
256  x  256                 1024                          85250 ns                   0.00127032 ns                         7.87201e+11
512  x  512                 1024                          65862 ns                  0.000245355 ns                         4.07573e+12
1024 x 1024                 1024                         126727 ns                  0.000118024 ns                         8.47287e+12
2048 x 2048                 1024                         290455 ns                  6.76268e-05 ns                          1.4787e+13
4096 x 4096                 1024                         921533 ns                  5.36403e-05 ns                         1.86427e+13
8192 x 8192                 1024                        3565102 ns                  5.18791e-05 ns                         1.92756e+13
```


### Modular Inversion


```bash
Inversion on F(2^64 - 2^32 + 1)

  dimension             iterations                        total                           per op                            ops/ sec
128  x  128                 1024                        5203766 ns                     0.310169 ns                         3.22405e+09
256  x  256                 1024                          82512 ns                   0.00122952 ns                         8.13322e+11
512  x  512                 1024                          77638 ns                  0.000289224 ns                         3.45753e+12
1024 x 1024                 1024                         118414 ns                  0.000110282 ns                         9.06769e+12
2048 x 2048                 1024                         274240 ns                  6.38515e-05 ns                         1.56613e+13
4096 x 4096                 1024                         910183 ns                  5.29796e-05 ns                         1.88752e+13
8192 x 8192                 1024                        3418749 ns                  4.97493e-05 ns                         2.01008e+13
```

### Modular Exponentiation

```bash
Exponentiation on F(2^64 - 2^32 + 1)

  dimension             iterations                        total                           per op                            ops/ sec
128  x  128                 1024                       11360031 ns                     0.677111 ns                         1.47686e+09
256  x  256                 1024                       25175474 ns                     0.375144 ns                         2.66564e+09
512  x  512                 1024                       99603174 ns                     0.371051 ns                         2.69505e+09
1024 x 1024                 1024                      397708984 ns                     0.370395 ns                         2.69982e+09
2048 x 2048                 1024                     1593074358 ns                     0.370917 ns                         2.69602e+09
4096 x 4096                 1024                     6359742073 ns                     0.370186 ns                         2.70135e+09
8192 x 8192                 1024                    25503631486 ns                     0.371127 ns                          2.6945e+09
```
