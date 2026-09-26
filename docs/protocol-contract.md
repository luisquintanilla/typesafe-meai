# Native protocol contract

This unofficial adapter targets Jev's native HTTP API, not general chat. The
offline fixtures establish local behavior only. **No live service requests have
been performed or authorized.**

## Evidence and discrepancies

Sources: [HTTP API](https://docs.typesafe.ai/api),
[advanced primitives](https://docs.typesafe.ai/primitives/advanced),
[SDK v0.6.0](https://github.com/typesafe-ai/typesafe-sdk-js/tree/v0.6.0/src),
commit `66880ccded6cb642dc1809620c2b108c33730214`.

| Shape | Documentation / pinned SDK evidence | Live |
|---|---|---|
| Authentication | SDK `client.ts` and transport tests construct `Authorization: Bearer <api-key>` | NOT RUN |
| Instructions string/object/array/null | Advanced primitives document all four; HTTP reference calls instructions required | NOT RUN |
| Omitted instructions | SDK optional fields remain omitted; distinct from explicit null | NOT RUN |
| Null state | SDK permits it; HTTP state description omits null | NOT RUN |
| Noul true/false structured/null entries | Advanced primitives and SDK permit them | NOT RUN |
| Entire Noul criteria null | SDK permits it; retained distinctly from omission | NOT RUN |
| Choice criteria | Nonempty local validation, maximum 255 documented; server minimum unresolved | NOT RUN |
| Score criteria | Ordered 2-10 levels; structured/null entries documented in advanced primitives | NOT RUN |
| Score legend | HTTP says strings; SDK `EntryType` preserves structured/null original criteria | NOT RUN |
| Models | GET `/v1/models` returns a `{ "models": [...] }` wrapper; aliases need not list every version | NOT RUN |

The internal native transport sends one explicit state value. The public provider
accepts ordinary messages: a single user TextContent stays a literal string,
and a single application/json DataContent becomes an owned parsed
string/object/array/null. Multiple messages/contents become a `conversation`
array preserving `role`, `author`, and ordered `contents` entries with
`type:"text"|"json"` and `value`. JSON-looking literal text stays text.
JevResultContent is projected as its actual JSON envelope, not a data URI.
Other JSON scalars, malformed JSON, tool protocol, image and audio contents
are rejected before cache/HTTP; they are not silently dropped.
UTF-8 bytes are decoded strictly: malformed byte sequences cannot silently become
replacement characters. Valid Unicode state is preserved.

Questions are JSON in `ChatOptions.AdditionalProperties["typesafe.v1.questions"]`.
IDs and labels are ordinal/case-sensitive. Duplicate IDs are rejected before
building a map. Undefined instructions/criteria mean omitted; an explicit
JsonElement null remains null. Questions and state own copies of JSON storage.
Only `type`, `instructions`, and `criteria` are allowed as top-level question
fields. Unknown fields fail rather than silently losing intent; arbitrary fields
inside structured instructions/descriptions remain application data.

The canonical response is the **entire native envelope** in owned
`JevResultContent : DataContent`, using integration-owned media type
`application/vnd.typesafe-meai.jev-result.v1+json` (not an official service media type).
Typed read-only `Answers`, `Json`, `Model` and usage properties decode these same
bytes; they are excluded from serialization. Decoding needs no prior question
snapshot. The provider separately validates correlation with submitted questions
before returning evidence, including cache hits. Standard model and usage mirror native values. Total usage is
input+output if absent. No response ID or timestamp is fabricated. Unknown
additive JSON fields survive; RawRepresentation is unnecessary. Native response
validation rejects missing/unknown/mismatched answers, duplicate properties,
invalid ranges/indices/token counts and invalid distributions. Absolute
sum-to-one tolerance is `1e-6`, without renormalization; its fitness for actual
service precision remains unverified.
The aggregate check uses decimal arithmetic (up to 28 decimal places) so both
serialized decimal tolerance endpoints are inclusive; it does not alter the
original JSON or individual answer values.

Native-only responses need no TextContent and may have empty `.Text`. Typed
responses contain exactly one projected JSON TextContent and adjacent
JevResultContent in the same last assistant message, which is what MEAI's real
`ChatResponse<T>` reads. Buffered response updates retain the same content.

## Transport

The registration helper creates one HttpClient resilience owner: 30-second total,
10-second attempt, two retries, exponential 500ms initial / 5s cap with jitter,
standard Retry-After support. Transient HTTP connection failures, 408, 429, and
5xx (including 529) may retry. Caller cancellation, 401, 422 and local/protocol
errors do not become confidence or success. `retry-after-ms` is not implemented.
Duplicate service processing/billing is possible; there is no exactly-once or
idempotency guarantee.
Standard Retry-After delta/date values use the pipeline's TimeProvider.
The package's built-in HTTP-date delay calculation used system time even with an
injected pipeline clock, so this small delay callback supplies the same clock;
backoff, jitter, attempt counts and timeout enforcement remain Polly-owned.

The bounded buffering handler is **inside** resilience, so response body reads
are inside the attempt/total budgets. Default maximum success/error body is
4 MiB, configurable. Failed attempt responses are disposed. Redirect following
is disabled on factory-created clients. API exceptions expose status and request
ID; body diagnostics require explicit access and are not in exception messages.

The handler reads into bounded storage once, then replaces the consumed source
content with a private byte-backed content owner. The transport still calls the
guard, but for that owned representation it checks the caller token, declared
length **and actual byte length**, then reuses the same array. It neither reads
the body again nor makes a final `ReadAsByteArrayAsync` copy on success. A stricter
later limit still fails, regardless of a misleading Content-Length or a consumed
public content stream. Arbitrary caller content cannot claim private ownership.
JevResultContent separately copies the bytes at its public ownership boundary.

`.NET 10`'s `HttpContent.LoadIntoBufferAsync(limit, token)` was evaluated but is
not a behavior-preserving replacement here. Its already-buffered fast path skips
limit/token checks, and [StreamContent's compatibility dispatch](https://github.com/dotnet/runtime/blob/v10.0.0/src/libraries/System.Net.Http/src/System/Net/Http/StreamContent.cs)
can route derived classes to the legacy serialization overload **without the
token**. The existing disposal-tracking-only subclass/stalled-stream tests exposed
this loss of cancellation. The adapter therefore retains its proven token-per-read
loop; it does not alter those fixtures, abandon a background copy with WaitAsync,
or add a second timeout/retry owner. Existing sanitized IO errors and disposal
semantics remain unchanged.

Direct construction borrows HttpClient by default; ownership transfer is explicit.
For direct construction the host must supply the same nonredirecting, bounded,
resilient configuration: a plain borrowed HttpClient does not magically acquire
the factory pipeline. Do not layer another retry decorator over HTTP resilience.
Both paths enforce the same explicit body-size limit, including unknown-length
bodies. With a plain borrowed HttpClient, the host must pass a cancellation
deadline covering the whole operation: `HttpClient.Timeout` with
`ResponseHeadersRead` alone only covers receipt of headers. Default 10s/30s
attempt/total budgets apply to the factory path, not arbitrary borrowed clients.

Live verification is a separate user-approved activity using synthetic data,
explicit model/request count/retry disclosure, and no credentials in source.

The isolated live project is skipped by default and excluded from `verify.ps1`.
After separate approval, the host can set `TYPESAFE_LIVE_APPROVAL` to
`SYNTHETIC-3-REQUESTS`, supply `TYPESAFE_API_KEY` through its environment, and run
the live project with `dotnet test`. This authorizes one models GET and two
synthetic inference batches (normal mixed support ticket; null/omitted/structured
SDK variants), pinned to `jev-1.13.0`. The configured policy can attempt each
request three times: at most nine HTTP attempts, six potentially billable POSTs.
No response bodies are persisted by these tests. They assert contract invariants,
not exact model probabilities or quality; compatibility failures remain failures.
Do not enable this environment variable in CI. These checks have **NOT RUN**.
