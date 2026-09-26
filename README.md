# TypeSafe / Jev on Microsoft.Extensions.AI

**Unofficial, private proof of concept. Not a Microsoft or TypeSafe SDK and not
production certified.** No license or public publishing decision has been made.
No live inference is performed by the offline proof.

Targets .NET 10 and pinned published MEAI 10.10.0. Jev is a constrained native
decision provider, not a conversational text generator. Noul, Choice and Score
answers retain their complete distributions, server confidences, fractional
indices, legends, model, usage and additive fields in **JevResultContent**, an
owned `DataContent` subtype with read-only typed answer views.

Use one decorated `IChatClient`, ordinary prompts/messages and standard MEAI:

```csharp
// Client settings: Policy = new(new TypeSafeBooleanPolicy(0.2, 0.8), 0.8).
// These illustrative thresholds are host policy, not a calibration claim.
ChatResponse<SupportDecision> response =
    await client.GetResponseAsync<SupportDecision>("Does this ticket need follow-up?");
JevResultContent evidence =
    response.Messages.Last().Contents.OfType<JevResultContent>().Single();

AIFunction preview = AIFunctionFactory.Create(
    (bool followup) => new { previewOnly = true, followup },
    "preview", "Preview the support follow-up decision.");
ChatResponse proposal = await client.GetResponseAsync(
    "Choose the support action.", new ChatOptions { Tools = [preview] });
// A proposal never invokes the function. Execution requires host authorization.

sealed class SupportDecision
{
    [System.ComponentModel.Description("Does this ticket need follow-up?")]
    public bool Followup { get; init; }
}
```

Normal finite DTOs use Boolean/string-enum properties and standard descriptions;
no special closed/required attributes, state wrapper, schema client or
response accessor is required. Numeric Score fields still need an explicit
rubric in optional client field configuration. There is no free-form generation.

Use `AddAIContentType<JevResultContent>(JevResultContent.TypeDiscriminator)` on a
**copy** of `AIJsonUtilities.DefaultOptions` when serializing responses yourself.
The keyed provider registration already configures its internal native cache.
See [serialization/reporting boundaries](docs/architecture.md).

```powershell
pwsh -File .\eng\verify.ps1 -Stage A -Offline
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- native
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- pipeline
pwsh -File .\eng\verify.ps1 -Stage B -Offline
pwsh -File .\eng\verify.ps1 -Stage C -Offline
pwsh -File .\eng\verify.ps1 -All -Offline
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- evaluate
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- tool
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- schema
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- route
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- select-tool
dotnet run --project .\samples\TypeSafe.MEAI.Sample -- dispatch
```

See [actual proof status](docs/proof-results.md),
[protocol and service discrepancies](docs/protocol-contract.md),
[capabilities](docs/capability-matrix.md), and
[architecture, privacy and dependency bootstrap](docs/architecture.md).
The exact schema subset and **illustrative, uncalibrated**
[decision policies](docs/decision-policies.md) are documented separately.
Live verification needs separate explicit consent and synthetic data; absence
of credentials never counts as successful live testing.

`evaluate` judges a **fixed scripted primary response** using a single native
question batch and also reports a Jev typed primary response using an explicit
portable evidence snapshot. Existing MEAI disk reporting and `aieval` generate the
HTML/JSON report offline. A deliberate failing CompositeEvaluator child remains
a visible diagnostic error, never a pass. `tool` uses real MEAI
UseFunctionInvocation middleware on a scripted conversational client; the bounded
function calls the decorated Jev client. No paid generative model is required.

`schema` exercises actual MEAI typed output and threshold-only cache reuse.
`route` uses the existing experimental router over scripted destinations.
`select-tool` performs selection then selected arguments with **zero invocation**;
`dispatch` uses standard tool-approval contents and real invocation middleware
to authorize/approve one harmless preview function. It never
asks Jev for a conversational continuation after execution.

`-All` also creates the two private local packages and runs a fresh package-only
consumer (native evidence/serialization, real typed DTO/primitive output,
the separate evaluator and standard one-shot tools).
External dependencies keep verified signatures and pinned content hashes. Only
the two locally produced unsigned artifacts use a narrowly isolated,
filesystem-only consumer exception; strict repository signature policy stays
enabled and global trust settings are unchanged. No package is published. Generated proof outputs remain ignored under
`artifacts`; the latest package receipt is `artifacts\package-proof\latest.txt`.
