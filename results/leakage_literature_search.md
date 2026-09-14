# Literature Search: Is the INCLUDE HuggingFace Split Leakage Reported?

**Date:** 2026-06-24  
**Supervisor instruction:** Before claiming the leakage is unreported, search Google Scholar
and arXiv for "INCLUDE dataset duplicate, split leakage, data quality" and check papers
that actually benchmark on INCLUDE (including INCLUDE paper itself and HA-GCN).

---

## What We Found

### INCLUDE paper (Sridhar et al., ACM MM 2020)
- ACM page is paywalled (403). HuggingFace page says only: "split into train and test
  splits as described in the paper." No mention of known issues or split overlap.
- HuggingFace distribution row counts: Train=3,820  Val=425  Test=1,010  (Total=5,255)
- The original paper reports ~4,287 unique videos across 263 classes / 15 signers.
- The extra ~968 rows (5,255 − 4,287) arise because the same video appears in
  multiple HuggingFace splits — this IS the overlap, but the HuggingFace page does
  not document it.

### HA-GCN (Song et al., Journal of Information and Intelligence, 2025)
- Paper found at ScienceDirect (doi:10.1145/S294971592400074X) but full text paywalled.
- Abstract: "extensive experiments on AUTSL and INCLUDE demonstrate outperformance
  with a significant margin."
- No public preprint found on arXiv. Their exact split protocol is NOT described in
  any accessible excerpt.
- We cannot confirm whether they used the HuggingFace split (leaked) or the original
  paper split (clean). Their 99.02% figure is suspicious given our clean-split result
  of 88.81% mean, but cannot be proven without their code/data.

### Other papers benchmarking on INCLUDE
- iSign (arXiv 2407.05404): references INCLUDE as a prior dataset, does NOT benchmark
  on it, no split discussion.
- CISLR (EMNLP 2022): different dataset entirely.
- No paper found that explicitly mentions train/test overlap or data leakage in the
  INCLUDE HuggingFace distribution.

---

## Conclusion

**The HuggingFace split contamination (28.4% test overlap with train) appears unreported**
in the publicly accessible literature as of June 2026.

- Zero papers found that mention this overlap.
- The HuggingFace dataset card does not document it.
- HA-GCN's split protocol is undocumented in any accessible source.

---

## How to Frame the Claim (supervisor guidance)

**Safe framing:** "To our knowledge, the 28.4% train/test overlap present in the
HuggingFace distribution of INCLUDE (ai4bharat/INCLUDE) has not been previously
documented. Methods reporting >93% accuracy on INCLUDE (e.g., HA-GCN, Song et al. 2025)
do not describe their split construction, leaving open whether their evaluation used the
contaminated HuggingFace parquet. We report a clean re-benchmark using the official
signer-dependent (SD) split after deduplication, and quantify the inflation as +6.74pp
using an identical model and training configuration."

**Do NOT say:** "all prior work is wrong" or "HA-GCN definitely used the leaked split."

---

## What Still Cannot Be Confirmed
- Whether HA-GCN used HuggingFace parquet or independently constructed clean splits.
- Whether the INCLUDE paper itself distributes clean splits separately from HuggingFace.
  (ACM DL paywalled; recommend checking with supervisor's institutional access.)
