# Synthetic regression depth discovery

Purpose: user requested a default make_regression pilot at requested depth50, then
feature sampling if needed to obtain deeper distinct-feature paths. All successful
probes use seed432, 30 rounds, CPU hist lossguide, max_leaves512, eta.05, base_score.5,
80/20 split, nthread16/core20–35. Generator defaults are100 rows/100 features/10
informative/no noise; only random_state is fixed for reproducibility. Later probes
explicitly increase n_samples to10000 while retaining other generator defaults.

probe_depth.py records exact commands, defaults, source/library/model hashes, raw
input data and split indices, trained models, feature-depth distributions, cover
mass and a witness deepest path. It cross-checks realized depths with an independent
JSON traversal (comparing leaf-depth multisets, irrespective of traversal order).
The initial-check directory records a failed order-sensitive validation attempt;
its source/results are not part of the successful summary.

Results: true default-data pilot reaches raw/distinct9/9. Feature sampling at100rows
only reaches12/12. With10000rows, levels1/.5/.25/.1 reach distinct10/11/13/19.
At .05/.02/.01: distinct22/28/31, with4.16%/14.30%/19.09% of cover beyond16.
These are intentionally varied tree structures, not a claim that the tuned data
configuration is the make_regression default. All results so far use one seed;
no new SHAP accuracy measurements or figure have been produced for these models.
summary.csv lists all11 successful probes. Further seed checks and a standard-rule
accuracy sweep remain separate steps.
