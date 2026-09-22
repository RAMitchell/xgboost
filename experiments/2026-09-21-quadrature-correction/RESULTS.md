# Corrected QuadratureTreeSHAP verification

Hardware: AMD Threadripper PRO 7975WX, NVIDIA RTX PRO 6000 Blackwell. Timing uses 32 CPU threads. These are fresh measurements, not V100 timings.

## Efficiency diagnostic (previous Figure 1 metric)

[Efficiency diagnostic (PDF)](artifacts/figure1-efficiency.pdf)

Completed requested depths: [4, 8, 12, 16, 24, 32, 48, 55, 64].

| Depth | Realized depth | TreeSHAP mean error | Q4 | Q6 | Q8 | Q16 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 8.614e-08 | 8.487e-08 | 8.433e-08 | 8.377e-08 | 8.571e-08 |
| 8 | 8 | 7.081e-08 | 7.610e-08 | 7.347e-08 | 8.077e-08 | 7.555e-08 |
| 12 | 12 | 1.048e-07 | 4.980e-06 | 8.146e-08 | 8.890e-08 | 8.306e-08 |
| 16 | 16 | 3.982e-07 | 1.210e-05 | 9.801e-08 | 1.020e-07 | 9.847e-08 |
| 24 | 24 | 6.191e-06 | 2.837e-05 | 1.153e-07 | 1.033e-07 | 9.881e-08 |
| 32 | 32 | 6.510e-04 | 4.250e-05 | 1.564e-07 | 9.470e-08 | 9.266e-08 |
| 48 | 48 | 3.516e-01 | 4.806e-05 | 1.953e-07 | 9.313e-08 | 9.068e-08 |
| 55 | 55 | 4.372e+00 | 4.819e-05 | 1.943e-07 | 9.156e-08 | 8.748e-08 |
| 64 | 55 | 4.372e+00 | 4.819e-05 | 1.943e-07 | 9.156e-08 | 8.748e-08 |

## Figure 1: maximum absolute feature error

[Corrected figure (PDF)](artifacts/figure1-corrected.pdf)

Maximum absolute difference over 512 images, 10 classes and 784 feature contributions (bias excluded), comparing native CPU float32 predictions with independent float64 Gauss–Legendre quadrature using ceil(maximum unique-feature depth / 2) points (minimum 2). Each reference is cross-checked with eight additional points. Exactness refers to polynomial integration in real arithmetic; float64 rounding remains.

| Depth | Realized depth | Reference points | TreeSHAP | Q4 | Q6 | Q8 | Q16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 2 | 3.091e-07 | 3.332e-07 | 3.166e-07 | 3.166e-07 | 3.827e-07 |
| 8 | 8 | 4 | 3.576e-07 | 3.599e-07 | 3.599e-07 | 3.094e-07 | 3.599e-07 |
| 12 | 12 | 6 | 3.957e-07 | 1.867e-05 | 7.054e-07 | 7.054e-07 | 7.054e-07 |
| 16 | 16 | 8 | 7.601e-07 | 6.711e-05 | 3.752e-07 | 3.824e-07 | 3.530e-07 |
| 24 | 24 | 12 | 2.364e-05 | 1.691e-04 | 1.365e-06 | 4.938e-07 | 3.882e-07 |
| 32 | 32 | 16 | 6.779e-03 | 2.174e-04 | 3.274e-06 | 3.775e-07 | 3.704e-07 |
| 48 | 48 | 24 | 2.632e+01 | 2.377e-04 | 6.586e-06 | 5.551e-07 | 4.359e-07 |
| 55 | 55 | 27 | 1.479e+02 | 2.384e-04 | 6.824e-06 | 7.172e-07 | 7.172e-07 |
| 64 | 55 | 27 | 1.479e+02 | 2.384e-04 | 6.824e-06 | 7.172e-07 | 7.172e-07 |

## Independent accuracy checks

| Model | Rows | Unique depth | Float64 Q8 max feature error | CPU Q8 max feature error | GPU Q8 max feature error | Reference cross-check |
| --- | --- | --- | --- | --- | --- | --- |
| benchmark-adult-large | 16 | 14 | 1.061e-13 | 2.523e-06 | 1.426e-05 | 1.661e-13 |
| benchmark-adult-small | 16 | 6 | 1.388e-16 | 1.096e-08 | 2.022e-08 | 1.388e-16 |
| benchmark-adult-sparse | 16 | 13 | 1.055e-14 | 1.988e-07 | 9.747e-07 | 9.936e-15 |
| benchmark-cal_housing-large | 16 | 8 | 2.305e-13 | 1.113e-06 | 6.539e-05 | 1.343e-13 |
| benchmark-cal_housing-small | 16 | 5 | 1.110e-16 | 1.465e-08 | 2.059e-08 | 1.388e-16 |
| benchmark-cal_housing-sparse | 16 | 8 | 5.940e-15 | 3.420e-07 | 1.350e-06 | 8.826e-15 |
| benchmark-covtype-large | 16 | 14 | 2.220e-16 | 3.696e-08 | 5.186e-08 | 3.331e-16 |
| benchmark-covtype-small | 16 | 6 | 1.110e-16 | 1.361e-07 | 1.685e-08 | 5.551e-17 |
| benchmark-covtype-sparse | 16 | 18 | 2.220e-16 | 3.526e-08 | 6.765e-08 | 2.220e-16 |
| benchmark-fashion_mnist-large | 16 | 16 | 0.000e+00 | 1.745e-06 | 4.441e-06 | 1.732e-14 |
| benchmark-fashion_mnist-small | 16 | 6 | 1.665e-16 | 1.941e-08 | 1.868e-08 | 2.220e-16 |
| benchmark-fashion_mnist-sparse | 16 | 48 | 1.855e-11 | 3.134e-07 | 4.326e-07 | 8.438e-15 |
| sweep-4 | 100 | 4 | 2.442e-15 | 2.419e-07 | 4.766e-07 | 2.442e-15 |
| sweep-8 | 100 | 8 | 3.553e-15 | 2.980e-07 | 9.250e-07 | 2.887e-15 |
| sweep-12 | 100 | 12 | 2.887e-15 | 3.591e-07 | 9.533e-07 | 3.664e-15 |
| sweep-16 | 100 | 16 | 0.000e+00 | 3.824e-07 | 6.640e-07 | 6.217e-15 |
| sweep-24 | 100 | 24 | 5.277e-11 | 3.500e-07 | 8.500e-07 | 5.329e-15 |
| sweep-32 | 100 | 32 | 6.563e-10 | 3.526e-07 | 6.304e-07 | 9.548e-15 |
| sweep-48 | 100 | 47 | 8.162e-09 | 4.589e-07 | 7.208e-07 | 8.660e-15 |
| sweep-55 | 100 | 53 | 1.019e-08 | 4.288e-07 | 3.878e-07 | 8.105e-15 |
| sweep-64 | 100 | 53 | 1.019e-08 | 4.288e-07 | 4.562e-07 | 8.105e-15 |

Pairwise comparisons use one complete feature-pair matrix per benchmark model.

| Model | Float64 Q8 max pair error | CPU Q8 max pair error | GPU Q8 max pair error |
| --- | --- | --- | --- |
| adult-large | 8.327e-14 | 3.129e-06 | 1.329e-05 |
| adult-small | 1.110e-16 | 6.330e-09 | 7.728e-09 |
| adult-sparse | 2.698e-14 | 8.786e-07 | 3.809e-07 |
| cal_housing-large | 4.891e-14 | 7.593e-06 | 1.982e-05 |
| cal_housing-small | 1.388e-17 | 1.898e-09 | 7.415e-09 |
| cal_housing-sparse | 1.160e-14 | 2.354e-07 | 3.225e-07 |
| covtype-large | 1.249e-16 | 8.908e-09 | 1.018e-08 |
| covtype-small | 6.245e-17 | 5.108e-09 | 1.256e-08 |
| covtype-sparse | 1.527e-16 | 8.794e-09 | 8.794e-09 |
| fashion_mnist-large | 0.000e+00 | 1.183e-06 | 2.293e-06 |
| fashion_mnist-small | 1.457e-16 | 4.901e-08 | 4.901e-08 |
| fashion_mnist-sparse | 2.325e-13 | 5.080e-07 | 5.080e-07 |

## Runtime tables

96/96 cases recorded. A timeout is the original 600-second per-case deadline.

Order 1, cpu: 0.86–10.35x speedup; median 2.58x across 12 completed pairs.

Order 1, cuda: 1.16–7.87x speedup; median 2.26x across 11 completed pairs.

Order 2, cpu: 2.75–51.08x speedup; median 14.34x across 10 completed pairs.

Order 2, cuda: 0.17–5.82x speedup; median 1.81x across 11 completed pairs.

### Order 1 (1000 rows)

| Model | CPU TreeSHAP (s) | CPU Q8 (s) | Speedup | GPU TreeSHAP (s) | GPU Q8 (s) | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| adult-large | 7.002117 | 1.868125 | 3.75x | 0.395171 | 0.186366 | 2.12x |
| adult-small | 0.002129 | 0.001833 | 1.16x | 0.004191 | 0.000539 | 7.77x |
| adult-sparse | 0.632541 | 0.213340 | 2.96x | 0.030403 | 0.017122 | 1.78x |
| cal_housing-large | 32.400064 | 15.553757 | 2.08x | 2.430691 | 1.073455 | 2.26x |
| cal_housing-small | 0.003697 | 0.004322 | 0.86x | 0.003848 | 0.000489 | 7.87x |
| cal_housing-sparse | 0.347933 | 0.216569 | 1.61x | 0.023316 | 0.013605 | 1.71x |
| covtype-large | 0.149621 | 0.031706 | 4.72x | 0.013006 | 0.011174 | 1.16x |
| covtype-small | 0.002004 | 0.001338 | 1.50x | 0.004389 | 0.000932 | 4.71x |
| covtype-sparse | 0.033013 | 0.006333 | 5.21x | 0.007496 | 0.002319 | 3.23x |
| fashion_mnist-large | 42.601099 | 4.866730 | 8.75x | 2.270862 | 0.769287 | 2.95x |
| fashion_mnist-small | 0.035226 | 0.016023 | 2.20x | 0.013794 | 0.007198 | 1.92x |
| fashion_mnist-sparse | 6.605343 | 0.637970 | 10.35x | unsupported depth | 0.113664 | — |

### Order 2 (100 rows)

| Model | CPU TreeSHAP (s) | CPU Q8 (s) | Speedup | GPU TreeSHAP (s) | GPU Q8 (s) | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| adult-large | 23.137399 | 1.446022 | 16.00x | 0.482663 | 0.158203 | 3.05x |
| adult-small | 0.008242 | 0.001333 | 6.18x | 0.004059 | 0.000698 | 5.82x |
| adult-sparse | 2.097956 | 0.138439 | 15.15x | 0.039747 | 0.015774 | 2.52x |
| cal_housing-large | 34.739556 | 8.467637 | 4.10x | 2.582774 | 0.941919 | 2.74x |
| cal_housing-small | 0.005112 | 0.001858 | 2.75x | 0.003477 | 0.000647 | 5.37x |
| cal_housing-sparse | 0.521532 | 0.117737 | 4.43x | 0.027131 | 0.016044 | 1.69x |
| covtype-large | 1.645835 | 0.032222 | 51.08x | 0.016447 | 0.097107 | 0.17x |
| covtype-small | 0.027359 | 0.002023 | 13.52x | 0.005381 | 0.006104 | 0.88x |
| covtype-sparse | 0.311001 | 0.007964 | 39.05x | 0.013277 | 0.015054 | 0.88x |
| fashion_mnist-large | timeout | 6.826097 | — | 5.224537 | 2.883060 | 1.81x |
| fashion_mnist-small | 5.998857 | 0.363535 | 16.50x | 0.453712 | 2.015774 | 0.23x |
| fashion_mnist-sparse | timeout | 1.293159 | — | unsupported depth | 2.151592 | — |


## Separate GPU accumulation limitation

CalHousing-large has unique-feature depth 8, so 8-point quadrature is mathematically exact. Its native GPU QuadratureTreeSHAP feature error reached 6.54e-05, compared with 2.05e-07 for GPUTreeSHAP. Splitting the same model into batches and summing their feature contributions in float64 reduces this error substantially:

| Trees per slice | Maximum feature error |
| --- | --- |
| 1000 | 6.545e-05 |
| 100 | 2.666e-06 |
| 10 | 2.790e-07 |

This supports cross-tree float32 accumulation as a major source of the deviation. The production algorithm was not changed for this diagnostic, and all timing tables use the standard implementation. Do not claim uniformly better native accuracy than TreeSHAP on all workloads.


## Provenance and limits

- See README.md for exact protocols and the distinction from the paper hardware.
- Original model caches were unavailable; benchmark ensembles were regenerated from the tracked training script.
- Float64 appendix-generating scripts were not found in the local checkout or tracked experiment history. The new checks do not establish provenance of the original appendix tables.
- Full measurements, all baseline accuracy errors, model hashes, timing samples and failure logs are retained under artifacts/.
- Python higher-order and TreeGrad-comparison scripts use standard Gauss–Legendre and were not rerun.
