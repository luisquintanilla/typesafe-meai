using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;

namespace TypeSafe.MEAI.Tests.Tools;

// Test-local host policy, not an alternative provider/executor. All execution and
// approval content handling is performed by the actual MEAI middleware.
internal sealed class OneShotHost : IDisposable
{
    private readonly IChatClient _executing;
    private readonly AIFunction _registered;
    private readonly ChatOptions _options;
    private readonly string _name;
    private readonly string _description;
    private readonly JsonElement _schema;
    private FunctionCallContent? _trustedCall;
    private JsonElement _trustedArguments;
    private string? _approvalId;
    private ChatMessage[]? _trustedHistory;
    private bool _started;
    private bool _consumed;

    internal Func<FunctionInvocationContext, CancellationToken, ValueTask<bool>> Authorize { get; set; } =
        (_, _) => ValueTask.FromResult(true);
    internal Func<bool> PolicyStillAllows { get; set; } = () => true;
    internal IServiceProvider? Services { get; set; }
    internal IDictionary<object, object?>? Context { get; set; }
    internal int Authorizations { get; private set; }
    internal int ProviderCalls { get; private set; }

    internal OneShotHost(IChatClient provider, AIFunction registered)
    {
        _registered = registered;
        _name = registered.Name;
        _description = registered.Description;
        _schema = registered.JsonSchema.Clone();
        _options = new() { Tools = [registered] };
        var capture = new CaptureClient(provider, response =>
        {
            ProviderCalls++;
            var call = response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>().SingleOrDefault();
            if (call is not null)
            {
                _trustedCall = new(call.CallId, call.Name, call.Arguments);
                _trustedArguments = JsonSerializer.SerializeToElement(call.Arguments);
            }
        });
        _executing = capture.AsBuilder().UseFunctionInvocation(configure: invocation =>
        {
            invocation.MaximumConsecutiveErrorsPerRequest = 0;
            invocation.FunctionInvoker = async (context, token) =>
            {
                Validate(context.CallContent);
                if (!ReferenceEquals(_registered, context.Function))
                    throw new InvalidOperationException("Unregistered outer function.");
                Authorizations++;
                if (!await Authorize(context, token)) throw new UnauthorizedAccessException("Host authorization denied.");
                token.ThrowIfCancellationRequested();
                Validate(context.CallContent);
                if (!JsonElement.DeepEquals(_trustedArguments, JsonSerializer.SerializeToElement(context.Arguments)))
                    throw new InvalidOperationException("Invocation arguments changed.");
                if (!PolicyStillAllows()) throw new InvalidOperationException("Host policy changed.");
                context.Arguments.Services = Services;
                if (Context is not null) context.Arguments.Context = Context;
                context.Terminate = true;
                return await context.Function.InvokeAsync(context.Arguments, token);
            };
        }).Build();
    }

    internal async Task<ChatResponse> StartAsync(CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        if (_started) throw new InvalidOperationException("One-shot host already started.");
        _started = true;
        var response = await _executing.GetResponseAsync(NativeFixture.Messages(), _options, token);
        var request = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().SingleOrDefault();
        if (request is not null)
        {
            _approvalId = request.RequestId;
            // Trusted storage is independent of the object sent to an approver.
            _trustedHistory = JsonSerializer.Deserialize<ChatResponse>(
                JsonSerializer.Serialize(response, NativeFixture.ContentJson()), NativeFixture.ContentJson())!.Messages.ToArray();
        }
        return response;
    }

    internal Task<ChatResponse> ResumeAsync(ToolApprovalResponseContent response, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        if (_consumed || _approvalId is null || response.RequestId != _approvalId || _trustedHistory is null)
            throw new InvalidOperationException("Unknown or replayed approval.");
        _consumed = true;
        if (response.ToolCall is not FunctionCallContent call) throw new InvalidOperationException("Unknown call.");
        Validate(call);
        // Denial terminates in the host: never hand it to unrestricted middleware.
        if (!response.Approved) throw new UnauthorizedAccessException("Host approval denied.");
        var trusted = _trustedHistory.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().Single();
        var messages = _trustedHistory.Append(new ChatMessage(ChatRole.User, [trusted.CreateResponse(true)])).ToArray();
        return _executing.GetResponseAsync(messages, _options, token);
    }

    private void Validate(FunctionCallContent call)
    {
        if (_trustedCall is null || call.CallId != _trustedCall.CallId || call.Name != _name ||
            call.InformationalOnly || !JsonElement.DeepEquals(_trustedArguments, JsonSerializer.SerializeToElement(call.Arguments)) ||
            _registered.Name != _name || _registered.Description != _description || !JsonElement.DeepEquals(_schema, _registered.JsonSchema))
            throw new InvalidOperationException("Trusted declaration, correlation or arguments changed.");
    }

    public void Dispose() => _executing.Dispose();

    private sealed class CaptureClient(IChatClient inner, Action<ChatResponse> capture) : DelegatingChatClient(inner)
    {
        public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            var result = await InnerClient.GetResponseAsync(messages, options, cancellationToken);
            capture(result);
            return result;
        }
    }
}
