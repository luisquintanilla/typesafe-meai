# Decision engineering proof harness

For learning how to consume a generator, start at [samples](../../samples/README.md).
This project is deliberately **not** a quickstart. It retains protocol fixtures,
public-stage parity, tensor exports, native cancellation/recovery, API inventory
and comparison instrumentation formerly in `samples\DecisionComparison`.

From the repository root:

```powershell
dotnet build eng\DecisionComparison.slnx -c Release
dotnet run --project eng\DecisionProof -c Release --no-build -- compare
dotnet run --project eng\DecisionProof -c Release --no-build -- inventory
```

`compare` uses recorded synthetic HTTP, not live Jev. `stages`, `provider julia`
and `provider qwen` require explicitly supplied local assets; `provider ollaya`
is live and opt-in. The detailed [design](../../docs/decision-generator-design.md)
and `eng\verify-decisions.ps1` describe the gates. Proof modes/algorithms are
retained; beginner `generate` dispatch was replaced by independent file apps.
The project and its lockfile remain project-based; no sample framework is added.

Jeeves is included in the public API inventory. Its source-grounded offline
protocol tests and the fresh **Jeeves-only package consumer** in
`eng\verify-decisions.ps1 -Offline -Packages` exercise the HTTP integration
without a server or native model stack. This is not a live GPU proof. Start at
the [Jeeves quickstart](../../samples/quickstarts/Jeeves/README.md) for application
code, or the [reasoning-budget scenario](../../samples/scenarios/ReasoningBudget/README.md)
for explicitly authorized service experiments.
