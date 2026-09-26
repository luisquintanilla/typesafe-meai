# Bounded decisions, routing and tools

Sample thresholds are **illustrative and uncalibrated**. Server confidence is not
a correctness guarantee; per-field acceptance is not a joint probability.

## Ordinary schemas and explicit policies

Use the decorated `IChatClient.GetResponseAsync<T>` directly. MEAI generates the
schema with the configured serializer; the provider does not duplicate CLR
reflection. Normal finite DTOs do not need special required/closed attributes:
the provider fills all known supported properties and emits no extras. This
satisfies optional/open finite schemas without promising arbitrary dictionaries.
String enums need the normal string-enum serializer configuration.

Set `TypeSafeClientOptions.Policy` once, for example
`new TypeSafeDecisionPolicy(new TypeSafeBooleanPolicy(0.2, 0.8), 0.8)`.
Optional `Fields` entries (`TypeSafeFieldOptions`) override individual semantics,
thresholds or validators using escaped JSON Pointers (`~0`, `~1`). Explicit
field instructions take precedence over schema descriptions, including local-ref
use-site descriptions. Otherwise questions identify the field path in the task
and conversation. `ChatOptions.Instructions` contributes shared semantics.
Response-format `SchemaDescription` and each ancestor object description are
also carried into affected questions as ordered `schemaDescriptions` entries
with `source`, escaped `pointer`, and `description`. Object-ref use-site
descriptions override target annotations without mutating the schema. Explicit
field instructions override only the leaf description, not this ancestor
context. Changing an effective description therefore changes native inference
and its cache key; threshold-only changes do not.
`DescriptionAttribute`, method/parameter descriptions, prompts and enum labels
are sufficient for ordinary Boolean/enum decisions; repetitive bindings are not
required. Enum labels may have null criterion descriptions.

Numbers still require explicit ordered rubric levels in `Fields`. A Score is
a fractional index into a 2-10-level rubric, never arbitrary quantity extraction.
An optional `ValidateResult` callback can reject the entire projected value.
Unused field overrides are rejected to catch misspelled paths; configure
different schema shapes separately when their overrides differ.

| Supported shape | Constraint |
|---|---|
| Finite object/nesting | Declared `properties`; output includes all supported known fields, no extras |
| Boolean | Noul; explicit inherited or per-field `falseMaximum < trueMinimum` |
| String enum | 1-255 distinct nonempty labels; explicit inherited/per-field confidence |
| Number | Explicit Score rubric and confidence, complete rubric range allowed by min/max |
| Local acyclic `$ref` / `$defs` | No external retrieval or recursion; preserve use-site annotations |
| Tool-only fixed scalars | Validate declared types/enums/exact numeric bounds before inference |

Keyword allowlist: `type`, `properties`, `required`, `additionalProperties`,
`enum`, `$ref`, `$defs`, `minimum`, `maximum`, `title`, `description`, `$comment`,
`examples`, `default`, `$schema`. The only accepted explicit dialect is
`https://json-schema.org/draft/2020-12/schema`. Annotation fields are not extra
acceptance rules. Ref assertion siblings, hidden unsupported assertions in
definitions, arrays, arbitrary strings/dictionaries, nullable/unions, inferred
integers, unbound numbers, recursion, pattern/format/multipleOf and composition
keywords reject with `TypeSafeSchemaException.Pointer` before HTTP.
Maximum traversal depth is 32; maximum inferred leaves is 256.

Numeric comparisons use exact bounded decimal-significand/exponent arithmetic,
not rounded double comparisons. The double check only bounds supported finite
range. Literal length over 1024 or explicit/effective decimal exponent outside
`[-1024,1024]` rejects. Numeric bounds must permit the **whole** rubric range;
Scores are never clamped or rounded. Mathematically integral fixed values `1.0`,
`1e2` and `9007199254740993.0` are canonicalized exactly before ordinary MEAI
integer parameter binding; `9007199254740992.5` remains nonintegral.

## Projection and evidence

For Noul `p`, accept false at `p <= falseMaximum`, true at `p >= trueMinimum`,
and abstain in the open interval. Bounds are finite with
`0 <= falseMaximum < trueMinimum <= 1`. Choice/Score use inclusive **server
confidence >= minimum**, not winning probability. Zero explicitly accepts all
valid confidences; there is no implicit accept-all.

No incomplete/default `T` is returned. `TypeSafeAbstentionException` carries
self-contained `Evidence` and path-specific `Rejections`; transport/schema/
protocol failures are not relabeled policy uncertainty. Successful typed output
puts projected JSON TextContent and JevResultContent in the **same last assistant
message**. MEAI wraps primitive roots with `data`, so overrides for a primitive
use `/data`. Streaming is buffered, not partial-object streaming.

Inference is cached **before** projection, so changing only thresholds/validators
can reuse native distributions. Changing field instructions, descriptions,
criteria/rubrics/order, task/state, effective model or provider/nonsecret account
scope changes inference and must miss. Evidence bytes are canonical; owned
`typesafe.v1.fields` metadata maps question IDs to pointers and
`typesafe.v1.batch-role` distinguishes native/projection/tool batches. Per-call
metadata is applied to a fresh content snapshot, never to shared cache evidence.

## Direct MEAI routing

The `route` sample directly composes pinned experimental `RoutingChatClient`.
Materialize one-shot histories **before** routing: classifier and destination
both consume them. Classification uses a separate request/options/model through
the decorated provider. Downstream options/model/usage remain the destination's;
the router does not dispose borrowed clients. Per-call evidence stays local.
The host explicitly chooses a confidence fallback or throws on abstention.
Do not catch transport/protocol failures as a fallback condition. Do not mutate
shared messages/options during a request. No competing public TypeSafe router
or hidden transport-error fallback is supplied.

## Standard tools, not a parallel registry

Supply ordinary `AIFunctionFactory.Create(...)` tools via `ChatOptions.Tools`.
The provider validates all candidate names/descriptions/schemas before inference,
including those not selected. It snapshots schema/configuration and verifies
declarations did not change before emitting a call. Maximum 254 unique functions;
`__typesafe_no_match__` is reserved.

Only required Boolean/enum arguments are inferred, including finite nesting.
`TypeSafeClientOptions.Tools[name]` optionally supplies `TypeSafeToolOptions`
for field overrides, validated fixed scalars, optional-default omission and
deterministic validation. It is configuration, not a special function wrapper.
Optional tool parameters require an explicit fixed value or omission with a
valid schema default. Injected services, credentials and host authorization
context must not be inferred arguments.

Auto selects one/no-match; RequireAny omits no-match; RequireSpecific skips the
selection batch. None validates declarations but emits no call/inference.
AllowMultipleToolCalls=true is rejected. Selection followed by **only selected**
arguments makes at most two native batches. No-match has evidence but no call;
uncertainty/invalid projection throws evidence-bearing abstention.
An accepted response has at most one FunctionCallContent, fresh per-request
correlation ID and one JevResultContent per native batch. Usage is summed once.
No side effect or approval is cached.

## One-shot host execution

The `dispatch` sample uses real `UseFunctionInvocation`, not a custom loop or
invoker. Its `FunctionInvoker` authorizes the actual registered **outer** function
and arguments, checks cancellation, consumes a one-shot trusted grant, sets
`context.Terminate=true`, and invokes `context.Function.InvokeAsync`. Set
`MaximumConsecutiveErrorsPerRequest=0`; a function error must not become a model
retry. An iteration limit alone does not prevent another model turn.

`ApprovalRequiredAIFunction` is a marker; standard middleware produces
`ToolApprovalRequestContent` and resumes with `ToolApprovalResponseContent`.
The host authenticates approval, validates correlation/name/arguments, prevents
replay and owns authorization. Denial/cancellation ends this bounded host
workflow **before** calling middleware again; feeding a denial into arbitrary
default middleware can cause another model turn. Preserve outer decorators and
the public approval marker contract. This is not global exactly-once delivery.

An optional Jev judge-as-tool is simply an `AIFunctionFactory` closure over the
decorated `IChatClient`, as shown by `tool`. Normal application functions have
no Jev dependency. No special factory, registry, selector or invoker public API
is necessary.
