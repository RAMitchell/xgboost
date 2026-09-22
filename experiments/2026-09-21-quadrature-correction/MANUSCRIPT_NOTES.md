# Manuscript changes supported by this rerun

1. Replace Figure 1 with `artifacts/figure1-corrected.pdf`. It uses the original
   displayed depth grid (4,8,12,16,24,32,48,55). The depth-64 extension is retained
   in the data and realizes depth 55; the depth-55 and depth-64 saved models
   are byte-identical. The metric is now maximum absolute per-feature difference
   from sufficient-order float64 quadrature over 512 images, 10 classes and
   784 features (excluding bias), replacing mean efficiency error. Update the
   caption and metric definition accordingly. Four points now reach the float32 rounding
   scale at depth 8, as the theorem predicts.
2. The mathematical exactness result is not contradicted by this finding.
   The corrected C++ rule now uses standard Gauss-Legendre nodes and weights on
   [0,1], matching the stated method. Keep the distinction between exact
   integration in real arithmetic and finite-precision implementation error.
3. Replace the affected C++ runtime tables with the completed new-host tables
   in RESULTS.md / artifacts/timings.csv, and update the hardware and thread
   configuration. These reruns use a Threadripper PRO 7975WX (32 threads) and
   RTX PRO 6000 Blackwell; do not retain the old Xeon/V100 caption or infer that
   runtime changes are caused solely by the quadrature correction.
4. Table 1 need not change: all 12 regenerated ensembles match its tree counts,
   realized maximum depth, unique-feature depth and mean leaves at the published
   precision. This is a structural match; the original model hashes were not
   available for a byte-for-byte comparison.
5. Qualify broad numerical-accuracy claims. The reproduced deep Fashion-MNIST
   stress experiment supports the stability advantage, but native GPU
   QuadratureTreeSHAP is not uniformly more accurate than GPUTreeSHAP. On the
   16 checked CalHousing-large rows, its largest feature error was 6.54e-05,
   compared with 2.05e-07 for GPUTreeSHAP. Both integrate a polynomial for which
   eight points suffice. Tree batching plus float64 summation reduces the
   QuadratureTreeSHAP error to 2.79e-07, implicating cross-tree accumulation.
   The production accumulator was not modified in this experiment.
6. The tracked Python scripts for Table 4, Figure 2 and the TreeGrad comparison
   already use standard Gauss-Legendre. This C++ bug does not require replacing
   those measurements. This is a code-path audit, not a new reproduction of
   those experiments.
7. Original Appendix A.2/A.3 generating scripts and source measurements could
   not be located. Do not claim their provenance has been verified. The new
   float64 results can be presented as a newly documented validation with the
   explicit models, depths, row counts and seed in README.md, or the original
   appendix scripts must be recovered to reproduce those exact tables.

Suggested replacement for the core numerical-stability observation:

> With the standard Gauss-Legendre rule, four points are sufficient for exact
> first-order integration at unique-feature depth eight. In our Fashion-MNIST
> sweep, Figure 1 reports maximum absolute per-feature error across all 512
> explained images and 10 output classes, relative to independent float64
> quadrature with ceil(d/2) points, where d is the maximum unique-feature path
> depth. Each reference is checked using eight additional points. This measures
> approximation and native float32 arithmetic error; exact polynomial integration
> does not eliminate finite-precision rounding. These results do not imply a
> universal error bound or uniformly better accuracy on every workload.


The exact runtime speedup claims should be taken from the completed rerun, not
from the previous paper ranges. Failed/unsupported/timed-out baseline cases
must remain visible rather than being included in a numerical speedup ratio.
