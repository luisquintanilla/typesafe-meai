# Azure OpenAI decisions with the proposed Microsoft.Extensions.AI APIs

This sample uses the decision APIs proposed in
[dotnet/extensions#7764](https://github.com/dotnet/extensions/issues/7764), rather than this
repository's own `DecisionInference` types:

- `IDecisionClient` and the decision request/answer types from `Microsoft.Extensions.AI.Abstractions`
  ([dotnet/extensions#7795](https://github.com/dotnet/extensions/pull/7795))
- typed `DecisionDefinition<TResult>` APIs from `Microsoft.Extensions.AI`
  ([dotnet/extensions#7796](https://github.com/dotnet/extensions/pull/7796))
- the `AsIDecisionClient` adapter from `Microsoft.Extensions.AI.OpenAI`
  ([dotnet/extensions#7957](https://github.com/dotnet/extensions/pull/7957))

The APIs are experimental, unreleased and may change. The packages are built locally from those pull requests.

It asks a decision model deployment to classify a support ticket twice: once from typed JSON state and
once from text state plus [an image](declined-payment.png) passed as contents.

## Build the packages

Clone dotnet/extensions with the typed-layer branch (#7796) and the OpenAI adapter branch (#7957)
checked out in two worktrees, run `restore.cmd` once, then from this repository's root:

```powershell
.\eng\build-official-decision-packages.ps1 `
    -TypedLayerCheckout <path-to-#7796-worktree> `
    -OpenAIAdapterCheckout <path-to-#7957-worktree>
```

The script checks that both worktrees contain the same `Microsoft.Extensions.AI.Abstractions` code and
writes `11.0.0-decisions.1` packages to the ignored `artifacts\official-feed` folder. [NuGet.config](../NuGet.config)
in `samples\official` reads that feed.

## Run

The sample needs an Azure OpenAI (Foundry) resource with a decision model deployment, such as
`gpt-6-luna`, and an identity that `DefaultAzureCredential` can use (for example, `az login`)
with access to it. No keys are used.

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://<resource>.cognitiveservices.azure.com"
$env:AZURE_OPENAI_DECISION_DEPLOYMENT = "gpt-6-luna"
dotnet run --file samples\official\AzureOpenAI\AzureOpenAI.cs
```

Example output:

```text
== Typed state (model gpt-6-luna)
Category: Account
  Billing    0.0%
  Technical  0.0%
  Account    100.0%
Refund probability: 0.0%
Expected urgency (0=low, 2=high): 1.73
Tokens: 435 in, 0 out
== Text + image (model gpt-6-luna)
Category: Billing
  Billing    100.0%
  Technical  0.0%
  Account    0.0%
Refund probability: 19.0%
Expected urgency (0=low, 2=high): 1.92
Tokens: 498 in, 0 out
```
