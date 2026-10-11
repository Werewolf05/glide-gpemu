# GLIDE Paper Revision Material

This document provides replacement material for the supplied PDF manuscript. The worktree contains the PDF but no TeX or Word source, so the PDF itself has not been edited. The replacement sections below incorporate the corrected scheduler experiment and distinguish it from the older results.

## Replace the HASP Score Paragraph

HASP assigns each pending request a score combining estimated compute affinity, memory fit, and an age-based priority boost. At scheduling time $t$, the score for request $r$ is

$$
S_r(t) = \frac{1}{C_r} \, M_r \, \left(1 + \frac{W_r(t)}{T_a}\right),
\tag{1}
$$

where $C_r$ is the estimated compute time in milliseconds, $M_r$ is the memory-fit factor, $W_r(t)=\max(0,t-a_r)$ is the time request $r$ has waited in seconds, $a_r$ is its arrival time, and $T_a$ is the aging threshold in seconds. In the default implementation, $M_r=1$ when estimated memory use is no greater than the configured memory threshold and $M_r=0.5$ otherwise. The default memory threshold is 2048 MB and the default aging threshold is 5 seconds. At each decision, HASP selects the pending request with the highest score. The inverse-compute term favors shorter jobs, while the aging factor progressively raises the priority of requests that have waited longer.

## Replace the Experimental Setup and Results Section

### IV. Results and Discussion

#### A. Experimental Setup

GLIDE was evaluated in two distinct emulator experiments. First, the saved GPU/model benchmark contains one batch-size-32 request for each of 36 models on six configured GPU profiles, for 216 combinations. The report marks all 216 combinations as PASS. These are profile-driven emulation outputs, not direct measurements on physical GPUs.

Second, FIFO, SJF, and HASP were compared using the corrected virtual-time discrete-event runner. The simulation models a single, non-preemptive service resource. Each request's service duration is taken from the selected GPU/model/batch profile, and each scheduler receives the same generated arrival trace within a trial. The verified validation uses the Tesla M40 profile, six models (ResNet-18, ResNet-50, VGG-16, MobileNet-v2, DenseNet-121, and AlexNet), batch sizes 1, 8, 16, and 32, Poisson arrivals at 40 requests per second, and a 15-second workload duration. Thirty trials use seeds 42 through 71. HASP uses its default 5-second aging threshold. Reported values are means across trials; per-policy 95% confidence intervals are available in the machine-readable report. The virtual clock respects future arrivals and advances completion by profiled service duration rather than wall-clock sleep.

The corrected runner allows each policy to select from all requests that have arrived, rather than restricting the scheduler to a one-request batch. The earlier scheduler aggregate files in `experiment_results/` were produced before this correction and should not be combined with or cited as results from the corrected comparison. The complete configuration and trial outcomes for the corrected run are saved in `experiment_results/hasp_validation_poisson40/report.json`.

#### B. GPU/Model Benchmark

Table 1 summarizes the saved GPU/model benchmark. Mean compute time is the arithmetic mean across the 36 model entries for each GPU profile. The fastest and slowest entries are the model-level compute times in the report.

**Table 1. Profile-driven GPU/model benchmark summary (batch size 32).**

| GPU profile | PASS | Mean compute (ms) | Fastest model (ms) | Slowest model (ms) |
|---|---:|---:|---|---|
| NVIDIA A100-SXM4-40GB | 36/36 | 17.84 | alexnet (1.70) | densenet201 (49.21) |
| Quadro RTX 6000 | 36/36 | 39.24 | alexnet (3.35) | resnext101_32x8d (132.85) |
| Tesla K80 | 36/36 | 218.22 | shufflenet_v2_x0_5 (15.21) | resnext101_32x8d (943.17) |
| Tesla M40 | 36/36 | 104.00 | alexnet (9.02) | resnext101_32x8d (353.50) |
| Tesla P100-PCIE-16GB | 36/36 | 57.96 | alexnet (5.08) | resnext101_32x8d (241.58) |
| Tesla V100-PCIE-32GB | 36/36 | 35.85 | alexnet (3.26) | resnext101_32x8d (131.17) |

PASS indicates that the report recorded positive compute and memory estimates for the combination; it does not establish timing accuracy against a physical GPU. The benchmark demonstrates coverage of the configured profile/model matrix.

#### C. Scheduler Comparison

Table 2 reports the verified 30-trial scheduler comparison. Entries are mean ± standard deviation across trials. Throughput is requests per second and latency values are milliseconds. The legacy Jain value is calculated from raw request latencies and measures equality of latency values, not low latency. New runs also report slowdown fairness, maximum wait, and service-relative wait outliers.

**Table 2. Scheduler results across 30 paired Poisson workload trials.**

| Scheduler | Mean latency (ms) | P95 latency (ms) | P99 latency (ms) | Maximum latency (ms) | Throughput (requests/s) | Jain latency index (legacy) | Relative outliers (legacy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 2204 ± 325 | 4000 ± 676 | 4187 ± 683 | 4251 ± 690 | 31.24 ± 0.11 | 0.773 ± 0.041 | 0.0 ± 0.0 |
| SJF | 808 ± 116 | 7498 ± 1019 | 11816 ± 1374 | 12808 ± 1450 | 31.24 ± 0.11 | 0.101 ± 0.006 | 48.3 ± 3.0 |
| HASP | 871 ± 128 | 5573 ± 718 | 7294 ± 866 | 7657 ± 866 | 31.24 ± 0.11 | 0.181 ± 0.007 | 82.2 ± 4.7 |

Relative to SJF, HASP increases mean latency by 63.7 ms, while reducing mean P95 latency by 1924.9 ms, mean P99 latency by 4522.5 ms, and mean maximum latency by 5150.3 ms. These are descriptive paired-trial differences; the saved report contains per-policy confidence intervals, but not stored paired-bootstrap intervals for these differences. Throughput is unchanged because all three policies process the same work on the same single-server service model. Thus, on this high-load workload, HASP improves the average-versus-tail-latency trade-off relative to SJF at a modest average-latency cost. It does not improve every metric.

HASP's results should not be described as better than FIFO on all fairness or tail measures. FIFO's mean P95 latency is lower than HASP's in this experiment, and its Jain latency index is higher. FIFO's mean latency is substantially higher, however. Jain's index here is computed on raw latencies, so it measures equality of latency values rather than whether those values are low; it should be interpreted alongside mean and tail latency.

The reported `starvation_count` is an older metric: it counts requests whose latency exceeds three times that run's mean latency. It is a relative outlier count, not a direct count of requests that waited indefinitely. By this metric, HASP's count is higher than SJF's (82.2 versus 48.3), so the present experiment does not support a claim that HASP reduces relative outliers. New experiments use maximum wait, slowdown fairness, and service-relative wait outliers based on queue wait versus three times profiled service time.

#### D. Limitations

The verified scheduler evaluation is one workload family at one configured arrival rate and one GPU profile in an emulator. Its 30 seeded trials support conclusions about that tested setup, not all workloads or physical GPU systems. Emulated service times are profile-derived rather than cycle-accurate hardware measurements. The implementation now supports a broader offline matrix of six GPU profiles, fifteen models, three batch sizes, three arrival patterns, and three load levels with paired traces and per-trial artifacts. That matrix is a reproducible protocol and runner capability; its numerical results must be generated before being cited. Physical-device measurements are required before drawing conclusions about absolute timing accuracy.

## Replace the Conclusion

This paper presented GLIDE, a profile-driven framework for studying deep-learning inference scheduling through GPU emulation. GLIDE combines layer-level profiling, model decomposition, workload generation, request queuing, batching, resource modeling, and scheduler evaluation. Its intended use is controlled comparative study; its emulated timings are not cycle-accurate predictions of physical GPU execution.

In a corrected 30-trial high-load Poisson evaluation, HASP achieved lower P95, P99, and maximum latency than SJF, with a modest increase in mean latency and no throughput change. Relative to SJF, HASP reduced mean P95 by approximately 26%, mean P99 by approximately 38%, and mean maximum latency by approximately 40%. The result demonstrates a workload-specific average-versus-tail trade-off rather than an across-the-board improvement. FIFO retained lower tail latency than HASP in this particular workload but had substantially higher average latency. The legacy relative-outlier count was higher for HASP and is not evidence of starvation freedom.

The next evaluation step is to execute the supported offline matrix and report results separately by GPU profile, model, load, and arrival pattern. The matrix should use frozen HASP parameters, paired traces, per-trial confidence intervals, slowdown fairness, maximum wait, and service-relative outliers. Physical-device measurements, controlled HASP ablations, and batching validation remain important future work.

## References and Citation Check

The supplied PDF contains a populated eight-entry reference list; it is not empty. Preserve that list when integrating these sections, and check every in-text citation against the final bibliography. The quantitative values above come from the local benchmark and repeated-experiment artifacts, not from the cited papers.

## Before Submission

- Keep the older scheduler result files separate from the corrected 30-trial comparison; do not mix results from the pre-correction and corrected runners.
- Do not cite the broader matrix until it has actually been executed; use the current 30-trial Tesla M40 result as the verified numerical evidence.
- Run the offline matrix with `run_publication_matrix(...)`, then add its aggregate values as separate tables by GPU profile rather than averaging incompatible profiles together.
- Replace legacy Jain/outlier columns with slowdown fairness, maximum wait, P99, and service-relative wait outliers for regenerated results.
- Rebuild the manuscript from its source and verify the equation, tables, and citations in the final PDF. The provided manuscript is a PDF export and no editable source was present in the worktree.