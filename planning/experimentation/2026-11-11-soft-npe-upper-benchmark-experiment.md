# Soft NPE upper benchmark experiment

See [execution and reproduction notes](../../experiments/soft-npe-upper-benchmark/README.md) for #185 and the mandatory adoption gate returning to #182.

The experimental CPU objective computes triangular pairs using sigmoid(-x) = 1 - sigmoid(x). This is the same all-pairs rank definition, with linear auxiliary memory and cached observation-side work; it does not truncate pairs or alter the approximation. On full-period HBV predictions across five softness values, agreement with the existing tensor loss was within 4.45e-16 absolute error.

The six-worker local pilot projected 9.37 hours excluding GA/startup/retries. Integrated full-budget single-basin trials took 3.94 s exact and 78.17 s at softness 0.1. These are feasibility measurements, not a completed equal-budget basin comparison or a VM runtime guarantee. The user subsequently requested the prior Hetzner CCX43 setup: 16 workers with one internal thread each. All scientific budgets and settings remain unchanged. Independent review and a VM timing pilot precede the full campaign; production adoption still requires explicit approval after results.
