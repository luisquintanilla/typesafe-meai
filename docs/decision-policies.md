# Bounded schema decisions, routing, and tools

All thresholds in the sample are **illustrative and uncalibrated**. A server
confidence is not a correctness guarantee. Accepting every field separately does
not establish a joint probability or calibrated correctness of an object.

## Generated schemas, explicit meaning

`TypeSafeSchemaChatClient` wraps the **decorated native client**, outside its
cache. Actual MEAI `GetResponseAsync<T>` generates the schema using the configured
serializer; the adapter does not reflect over CLR properties. Use required DTO
properties, closed objects, and string-enum serialization. The executable
`schema` sample uses all three native primitives and a primitive Boolean root.
MEAI wraps primitive roots in a required closed `data` object, so their binding
pointer is `/data`, not the empty root pointer.

Bindings use escaped JSON Pointers (`~0` for `~`, `~1` for `/`). They specify
meaningful question instructions, Boolean policy or explicit confidence, enum
descriptions, and a numeric rubric where applicable. A nonempty schema
`description` is the only instruction fallback. Field names alone do not define
judgment semantics. `ChatOptions.Instructions` on this adapter becomes shared
instructions in every native question; it is never an ignored wire option.

| Supported shape | Required semantics |
|---|---|
| Closed required object, including nested objects | `properties`, `required`, `additionalProperties:false` |
| Boolean | Noul with explicit `falseMaximum < trueMinimum` |
| String enum (1-255 distinct nonempty labels) | Choice with exact label descriptions and explicit minimum server confidence |
| Number | Score with 2-10 ordered rubric levels and explicit minimum confidence; **fractional rubric index**, not arbitrary quantity extraction |
| Local acyclic `$ref` / `$defs` | Resolved in this schema; no external fetch or recursion |
| Tool-only fixed scalar values | Validated string, Boolean, finite number/integer, enum and supported inclusive numeric constraints |

The exact keyword allowlist is `type`, `properties`, `required`,
`additionalProperties`, `enum`, `$ref`, `$defs`, `minimum`, `maximum`, `title`,
`description`, `$comment`, `examples`, `default`, `$schema`. `$schema`, if
present, must be `https://json-schema.org/draft/2020-12/schema`. Annotation fields
are not extra acceptance rules. `$ref` assertion siblings are rejected.
All other keywords are rejected, including unknown assertions hidden in
definitions. Maximum schema/object/reference traversal depth is 32; maximum
inferred decision leaves is 256. Question IDs follow traversal order (`q0000`,
etc.), so property names and escaped paths cannot collide.

Arrays, nullable/union schemas, arbitrary strings, inferred integers, open
objects, unbound numbers, optional inference, recursive/external references,
`pattern`, `format`, `multipleOf`, conditional/composition keywords, and other
unsupported assertions fail with `TypeSafeSchemaException.Pointer` **before
HTTP**. Numeric bounds must accommodate the entire rubric index range; the
adapter never rounds, rescales, clamps, or truncates a Score. There is no guessed
character-to-token limit; service validation errors remain explicit.
Numeric constraints and fixed integer-ness use exact decimal-significand/exponent
comparison, including values above `2^53`. The finite-double range check is not
used for comparison. Numeric literals longer than 1024 characters or with an
explicit/effective decimal exponent outside `[-1024,1024]` are rejected with a
pointer, not silently rounded. Reference use-site description/default annotations
override target annotations; explicit binding instructions still take precedence.
Mathematically integral fixed values such as `1.0` and `1e2` are canonicalized
with exact integer arithmetic to `1` and `100` before forming function arguments.
This makes them compatible with real MEAI integer-parameter deserialization
without rounding, including plain/fractional-notation integers above `2^53`.

## Projection and abstention

For Noul probability `p`, false is accepted when `p <= falseMaximum`; true is
accepted when `p >= trueMinimum`; the **open interval between** abstains.
Bounds must be finite and satisfy `0 <= falseMaximum < trueMinimum <= 1`.
Choice/Score accept when **server confidence >= configured minimum**, not when
the winning probability exceeds that minimum. Zero explicitly accepts any
valid confidence; there is no implicit accept-all policy.

Per-field validators and an optional whole-object validator can reject an
otherwise accepted projection. The adapter returns no incomplete/default `T`.
Actual MEAI `GetResponseAsync<T>` throws `TypeSafeAbstentionException` containing
full evidence and path-specific rejection reasons; `GetTypeSafeDecisionAsync<T>`
returns accepted/abstained status explicitly. Accessing `Value` on an abstained
decision fails. Transport/protocol/schema errors are **not** converted into
policy abstention.

Successful projected responses retain the complete native envelope and
questions in serializable `AdditionalProperties` (`typesafe.v1.evidence`,
`typesafe.v1.evidence.questions`). `GetTypeSafeEvidence()` decodes them after
normal MEAI serialization. `RawRepresentation` is never needed. Streaming is
buffered and uses standard response updates, not partial typed objects.

Thresholds and validators run **after** native caching. Changing a threshold
can change acceptance without another native call. Changing instructions,
enum descriptions, rubric levels/order, state, effective model or provider/
tenant scope changes inference and must not reuse stale evidence. See the
`schema` sample for an actual cached-response/changed-threshold demonstration.

## Stateless routing

`TypeSafeRoutingChatClient` uses the existing experimental **MEAI 10.10.0
RoutingChatClient**, with a localized diagnostic suppression. It borrows the
classifier, target pipelines and optional fallback; disposing it does not
dispose those application-owned clients.

Supply 1-255 unique route IDs with meaningful descriptions and an explicit
minimum server-confidence threshold. Classification uses one separate Choice
request through the decorated native pipeline. Its optional `classifierModelId`
is independent of the downstream request's `ModelId`. The default projector
preserves message role/order/author and text content in structured state; other
modalities require an explicit host state projector. Server-side conversation
and continuation handles are rejected before classification.

The router materializes a one-shot incoming message sequence once. MEAI clones
downstream options; classification never inserts its model or questions into
them. Do not concurrently mutate shared message objects or nested downstream
options while a routing call is active. One accepted target (or explicit
confidence fallback) receives one call and its response/model/usage are returned
unchanged. Without fallback, uncertainty throws an evidence-carrying routing
abstention. HTTP/protocol failures propagate and never select fallback.
An awaited per-call observation hook replaces mutable `LastDecision` state.

## Selection is not execution

`TypeSafeToolDefinition` snapshots and validates the original registered
`AIFunction.JsonSchema` with the same compiler. `TypeSafeToolSelector` accepts
1-254 unique functions, reserving the 255th Choice slot for explicit no-match.
All candidates must be valid before any inference, not only the eventual winner.

Only required Boolean/string-enum arguments are inferred, including closed
nested objects. Host-fixed scalar arguments are validated against their declared
type, enum and supported ranges. Unsupported assertions remain errors even when
the host supplies the value. Optional parameters require either an explicit fixed
value or explicit omission with a valid declared default; nothing silently
changes optional to required. Use MEAI's schema generation
`TransformOptions.DisallowAdditionalProperties = true`; do not use
`RequireAllProperties` to disguise optional parameters.

The two-step protocol makes **at most two logical native requests**:

1. Select one function or no-match, applying explicit selection confidence.
2. Only for accepted selection, infer that function's unresolved arguments in
   one batch (zero extra requests when everything is host-fixed/defaulted).

All invocation counters remain zero during selection. No-match/uncertain or
application-invalid arguments yield no executable call. Accepted selection
retains the original registered function, complete evidence for both batches,
and one `FunctionCallContent` with a unique correlation ID.

`TypeSafeToolInvoker.InvokeAsync` is a **separate host action**. It checks the
selected function/declaration/argument binding, awaits fresh host authorization,
requires a distinct affirmative approval callback for an
`ApprovalRequiredAIFunction`, checks again, then invokes the **original outer
function** with copied `AIFunctionArguments`, cancellation, context and services.
The public `GetService` approval marker is recognized through standard
`DelegatingAIFunction` wrappers; it is never used to unwrap for invocation.
Custom wrappers must preserve that public marker contract or enforce their own
approval mechanism; arbitrary hidden host policies cannot be discovered.

Denied approval, changed binding, invalid arguments or cancellation invoke
nothing. Function exceptions propagate without retries. At most once applies
to **one explicit helper call**, not global replay prevention/exactly-once
delivery. The host owns replay prevention and must not mutate shared function
declarations or call objects concurrently with authorization/invocation.
There is no Jev-backed agent loop or synthesized final response. `dispatch`
uses a harmless synthetic preview function; model output is never authorization.
