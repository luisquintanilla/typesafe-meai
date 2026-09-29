# 2. Read the answer before you trust it

[Learning path](README.md) | Previous: [the first call](01-from-rules-to-decisions.md)
| Next: [deployment and evaluation](03-running-responsibly.md)

All numbers below are **worked illustrations**, not benchmark or live-service
results. The sample READMEs separately identify captured native output.

## Binary: one proposition, two outcomes

Ask "Does the ticket report a payment problem?" This is one **Binary** question.
`BinaryDecisionAnswer.TrueProbability` might be `0.7`; the complementary false
probability is `1 - 0.7 = 0.3`. The API intentionally does not return a Boolean
decision with a baked-in application threshold.

You supply meaningful instructions or meaningful true/false descriptions; IDs
are not used as missing instructions. Julia's local adapter requires paired
descriptions when descriptions are provided. Provider-specific preparation
rules can be stricter than the basic portable type.

Two independent Binary questions are **not** two entries in one multiclass
distribution. "Payment problem?" at 0.7 and "Customer mentions a password?"
at 0.6 need not sum to 1: a ticket can mention both. Do not normalize those
unrelated probabilities together.

## Choice: one of the offered alternatives

A Choice question returns `SelectedCandidateId` and a complete
`Probabilities` dictionary keyed by candidate ID:

| ID | Probability |
|---|---:|
| account | 0.70 |
| billing | 0.10 |
| technical | 0.20 |

These alternatives are interpreted as mutually exclusive choices **for this
question**. The selected ID must be one of them. Supplied IDs must be nonblank,
unique within the question and exact: no case folding, trimming or punctuation
rewriting. Descriptions carry their meanings.

The result validates question/answer correspondence, all candidates and valid
distributions; it is not just whatever JSON the model happened to emit. Portable
collections are read-only snapshots. Provider-specific raw evidence and mutable
extension properties are a different, advanced surface and are not a portable
business rule.

**A valid distribution is not the same as a correct answer.** Ask an unrelated
weather question with only support teams as candidates and the model still has
to favor something. It can be confidently wrong. It does not inherently return
"none of the above" unless you supply a described candidate, and even then
selecting that candidate is a learned behavior, not guaranteed abstention.

Adding a fourth candidate changes the question and the normalization. Rewording
"technical" to overlap with account access also changes the task. Comparing a
probability before and after that change is not comparing an unchanged event.

## Score: an ordered rubric, not an arbitrary number

A Score question describes ordered levels, for example:

| Zero-based index | Rubric description | Probability |
|---:|---|---:|
| 0 | No workflow disruption | 0.10 |
| 1 | A workaround is available | 0.20 |
| 2 | Work is blocked | 0.70 |

`ScoreDecisionAnswer.Probabilities` is in that rubric order.
`ExpectedIndex` is the weighted average of **indices**:

```text
0 * 0.10 + 1 * 0.20 + 2 * 0.70 = 1.60
```

It is not a probability, a 1-5 star rating, a dollar loss, or a claim that severity
is exactly 1.6. Index 1.6 need not name any actual rubric level. The most likely
level here is 2, which is a different statistic from the mean. The ordinal
descriptions establish order, not equal real-world distances. If you assign
costs or convert indices into a business scale, that is explicit application
policy requiring justification.

The API retains `NativeScore` separately when supplied. Bounds named
`ExpectedIndexLowerBound`/`UpperBound` describe numerical feasibility under
validation/rounding rules, **not confidence intervals or accuracy guarantees**.
Qwen's adapter supports **Choice only**, not Binary or Score.

## Tokens and budgets explain many rejected calls

A tokenizer maps text into numbered chunks called **tokens**. One word can
be multiple tokens; punctuation, Unicode and whitespace can matter. The
question framing, instructions, candidate descriptions and state all occupy
the model's **context**, its bounded input capacity.

The tokenizer must match the model's learned vocabulary. Swapping an unrelated
tokenizer can turn a perfectly valid C# string into the wrong model input.
A character count cannot reliably predict a token budget.

The local adapters reject state/instructions/options that exceed their budgets
before inference rather than silently chopping text or rewriting marker tokens.
Do not fix a long ticket by dropping the end without understanding what evidence
you removed. Design an explicit data-selection strategy and evaluate its losses.
Julia's and Laya's limits differ; an interface does not erase them.

## Optional deeper look: logits and softmax

A **logit** is a raw numeric model score. It is not itself a probability:
it may be negative, positive, or larger than 1. A common conversion is
**softmax**, which makes positive weights and divides by their sum.

Suppose three logits are `2, 1, 0`. Subtract the maximum first for numerical
stability: `0, -1, -2`. Exponentiate to get approximately `1, 0.3679, 0.1353`.
Their sum is `1.5032`; dividing each by it gives approximately
`0.6652, 0.2447, 0.0900` (rounded). The largest score gets the most probability,
but all three still contribute.

Subtracting the same constant does not change softmax. A provider may also use
model-specific temperature/scoring rules, so never compare raw logits across
different models as if they were the same scale. Laya's separate action head,
Jev confidence metadata and Qwen selected-token logits are distinct evidence.

For Qwen, the adapter scores only verified answer aliases A..Z and normalizes
that subset, **not the whole vocabulary**. A strong relative answer score
does not prove the unconstrained text model wanted to answer that way.

## Precision, rounding and "confidence" are different questions

The apps format numbers with `F4`. That rounds for human display only.
`1.0000` need not be exact certainty; `0.0000` need not mean impossible.
Printed rows may sum to 0.9999 or 1.0001.

Separately, `DecisionPrecision` records the provider's **reported numerical
representation**. High-precision distributions are validated with the contract's
absolute `1e-6` mass tolerance. Four-decimal observations (used by the supported
Ollaya protocol) must be compatible with some underlying normalized distribution.
They are not silently renormalized to invent precision.

For a rounded Score, `ExpectedIndex` is the weighted sum of the **observed**
probabilities and can marginally exceed the rubric range. The feasible latent
bounds describe what normalized unrounded probabilities could imply, not a
statistical confidence interval. Ordinary display rounding does not change the
underlying precision tag.

When someone says "confidence", ask what they mean:
the largest candidate probability, a provider-specific head, a numerical
interval, or observed real-world accuracy? None can be silently substituted
for the others.

**Calibration** asks whether a stated probability corresponds to empirical
frequency on comparable labeled cases. If a system is well calibrated for a
defined task, predictions near 0.7 would be correct about 70% of the time in
that population. A single output cannot establish calibration. Neither softmax,
temperature settings nor a vendor label prove it for your data.

## Practice reading, not just accepting

Predict the distribution for a clear, ambiguous and out-of-scope ticket before
running a sample. Observe which candidates compete. Change one description,
keep IDs fixed and run again. Explain why a result changed, or why you cannot
tell from one case. Do not interpret a refusal to fit context as uncertainty:
it is a rejected call with no prediction.

Next, design an [application policy and evaluation](03-running-responsibly.md)
instead of picking an arbitrary confidence cutoff.
