# Capabilities

| Native capability | Behavior |
|---|---|
| ModelId | Supported, default materialized before cache |
| Explicit state + native questions | Required |
| Plain JSON format | Supported |
| JSON Schema format | Native rejects; bounded Stage C adapter only |
| ChatOptions.Instructions | Native rejects; use explicit question instructions |
| Temperature, TopP/K, penalties, Seed, MaxOutputTokens | Reject explicit values |
| Tools, ToolMode, AllowMultipleToolCalls, StopSequences | Reject, including empty collections |
| Reasoning, ConversationId, ContinuationToken, AllowBackgroundResponses | Reject |
| RawRepresentationFactory | Reject even on a primed cache hit |
| Unknown nonreserved extension metadata | Owned serializable snapshot; not forwarded as wire options |
| Unknown `typesafe.*` options | Reject |
| Streaming | Full response buffered, then standard response updates; not token streaming |

The pinned `ChatOptions` surface is audited rather than assuming serialization
covers every option. Validation precedes both HTTP and cache. No hidden
last-message selection, general generation, multimodality, or native tool loop.

| Composition surface | Supported boundary |
|---|---|
| MEAI generic typed output | Real schema-format `GetResponseAsync<T>` over bounded schema adapter; no prompt-appending mode |
| Structured output | Required closed nested objects, bool, string enums, explicitly rubric-bound fractional numbers |
| Uncertainty | Explicit decision or evidence-carrying abstention exception; no incomplete/default DTO |
| Routing | Existing experimental router, separate classifier model, explicit confidence fallback/abstention, borrowed targets |
| Tool selection | Up to 254 registered functions + no-match; select first then infer only selected bool/enum arguments |
| Other tool arguments | Supported validated host-fixed scalars; explicit optional-default omission |
| Tool execution | Separate host-authorized one-shot outer-function call, distinct required approval, no automatic side-effect retry |

Full keyword allowlist, depth/leaf bounds, acceptance intervals, evidence
metadata and unsupported schema details: [decision policies](decision-policies.md).
These local capabilities do not resolve the [unverified service contract
variants](protocol-contract.md).
