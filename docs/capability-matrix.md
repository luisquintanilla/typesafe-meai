# Capabilities

The single public `TypeSafeChatClient` implements ordinary `IChatClient`. It is a
finite decision provider, not a general generator. Native cache entries sit
below typed policy projection and tool-call creation.

| Request surface | Behavior |
|---|---|
| ModelId | Supported; effective default materialized before native cache |
| Native questions | Serializable object at `AdditionalProperties[TypeSafeChatClient.QuestionsKey]` |
| Actual MEAI `GetResponseAsync<T>` | Schema-format mode; no special adapter, state wrapper or result accessor |
| `ChatOptions.Instructions` | Deliberately incorporated into native question semantics |
| Text/multiple messages | Literal text; explicit role/order/author/content boundaries in conversation state |
| JSON DataContent / JevResultContent | Owned strict UTF-8 JSON values, never opaque data URIs in inference state |
| Ordinary finite DTO | Known Boolean/enum/nested fields emitted even when open/optional; no extras |
| Numeric decision | Explicit Score rubric required; fractional index, not arbitrary extraction |
| `ChatOptions.Tools` | Existing AIFunctionDeclaration/AIFunction schemas, all candidates preflighted |
| Tool Auto/default | Select one or no-match; at most one call |
| Tool RequireAny | Selection without a no-match label; uncertainty throws abstention |
| Tool RequireSpecific | Supplied exact name required; skips selection inference |
| Tool None | Validates candidates, emits no call and makes zero native requests |
| AllowMultipleToolCalls | null/false supported; true rejected |
| Streaming | Complete buffered response followed by existing MEAI response updates |
| Unknown nonreserved metadata | Owned serializable snapshot; not forwarded as service options |
| Unknown `typesafe.*` metadata | Rejected |

These combinations fail explicitly **before native cache and HTTP**:

- No decision operation, or ambiguous native/schema/tool operations.
- Prompt-appended typed mode (`useJsonSchemaResponseFormat:false`), text format,
  arbitrary dictionaries/arrays/free text, nullables/unions, and unbound numbers.
- Temperature, TopP/K, penalties, Seed, MaxOutputTokens, StopSequences
  (including empty), Reasoning, ConversationId, ContinuationToken,
  AllowBackgroundResponses and RawRepresentationFactory.
- Unsupported modalities/tool protocol content. Host approval middleware consumes
  approval input; denial/cancellation terminates the host workflow before Jev.

| Composition | Boundary |
|---|---|
| Evidence serialization | Register subtype on copied options for AIContent **and inherited DataContent** polymorphism |
| Evaluation | Existing IEvaluator; one batch for configured metrics, Noul/Score numeric, Choice string |
| Disk reporting | Explicit host snapshot of Jev content to ordinary DataContent; no global serializer patch |
| Routing | Direct existing experimental RoutingChatClient; separate classifier options and explicit confidence fallback |
| Selection | Up to 254 unique functions, at most two batches, no provider-side invocation |
| Execution | Existing UseFunctionInvocation + host authorization, standard approval, Terminate=true, zero error recovery |

See [policies](decision-policies.md) for the exact schema subset and
[protocol discrepancies](protocol-contract.md) for still-unverified service variants.
