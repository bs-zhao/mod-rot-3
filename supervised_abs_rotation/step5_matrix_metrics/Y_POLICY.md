# Which Y can support final empirical success?

User clarification, 2026-11-12: final success must use an actual observed
choice, or a neural network's logit, probability, or choice output. A true
analytic/generating-equation logit or probability may assist discovery but
must never count as final success.

| Case Y.npz key | Source | Eligible final-validation Y type |
|---|---|---|
| choice_observed | Original observed 0/1 choice | Yes |
| logit_nn | Frozen NN class-1 score minus class-0 score | Yes |
| p_nn | Frozen NN two-class softmax probability | Yes |
| choice_nn | Frozen NN argmax, ties to class 0 | Yes |
| Analytic logit or probability | Known generating equation | Auxiliary only |

This is an input-source rule, not a claim that a particular metric has
succeeded. Synthetic unit-test fixtures verify code and are not real-data
validation. Current large-scale response banks use logit_nn and p_nn. The
other two permitted choices are present in the case files but have not
received the same new-wave evaluation; no results for them are implied.

`validation_response_policy.py` defines the final-channel types. The
matrix loader accepts these four channels. The continuous ledger labels
Y provenance and source eligibility, and final candidate reports exclude
analytic/unrecognized channels. Auxiliary experiments must remain labeled
as auxiliary even if their rotations look favorable.

`audit_response_provenance.py` verifies the actual four case Y files against
their recorded export hashes. On every original trial row it checks stored
NN logit/probability/choice against the stored two-class NN scores and
checks binary choice fields. It also records the response banks' channels.
The exporter source explicitly calls the frozen NN decoder for these fields
and preserves observed choice separately. The audit reads existing matrices
and metadata; it does not call a decoder/generator or read its weights.

All four actual audits pass: N=5000 each; CR3 has 2,500,000 original trial
rows and C1/C2/C3 have 2,003,448 each. The actual case Y files contain the
four allowed fields plus the stored two-class NN scores, with no analytic
logit/probability fields. Existing score artifacts are not rewritten.

```cmd
python supervised_abs_rotation/step5_matrix_metrics/audit_response_provenance.py
```

The audit is saved under
`outputs/supervised_abs_rotation/final_Y_provenance_audit/`. Its source
qualification remains separate from approximation quality, sparse-support
behavior, four-model comparisons and blind-optimization evidence.
