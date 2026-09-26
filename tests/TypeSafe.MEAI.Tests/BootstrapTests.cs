using Microsoft.Extensions.AI;
using Xunit;

namespace TypeSafe.MEAI.Tests;

public sealed class BootstrapTests
{
    [Fact]
    public async Task RoutingChatClient_Create_SelectsAndInvokesDownstreamExactlyOnce()
    {
        ChatMessage[] messages =
        [
            new(ChatRole.System, "p0-system"),
            new(ChatRole.User, "p0-user")
        ];
        var options = new ChatOptions
        {
            ModelId = "p0-downstream",
            Temperature = 0.25f
        };
        var originalModelId = options.ModelId;
        var originalTemperature = options.Temperature;
        using var cancellationSource =
            CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var cancellationToken = cancellationSource.Token;
        var expectedResponse = new ChatResponse(new ChatMessage(ChatRole.Assistant, "p0-routed-response"))
        {
            ResponseId = "p0-response-17"
        };
        using var downstream = new RecordingChatClient(expectedResponse);
        var selectorCallCount = 0;
        ChatMessage[]? selectorMessages = null;
        ChatOptions? selectorOptions = null;
        CancellationToken selectorCancellationToken = default;

#pragma warning disable MEAI001
        using IChatClient router = RoutingChatClient.Create((context, token) =>
        {
            selectorCallCount++;
            selectorMessages = context.Messages.ToArray();
            selectorOptions = context.ChatOptions;
            selectorCancellationToken = token;
            return new ValueTask<IChatClient>(downstream);
        });
#pragma warning restore MEAI001

        var response = await router.GetResponseAsync(messages, options, cancellationToken);

        Assert.Equal(1, selectorCallCount);
        Assert.Equal(1, downstream.ResponseCallCount);
        Assert.Same(expectedResponse, response);
        Assert.Equal("p0-routed-response", response.Text);
        Assert.Equal("p0-response-17", response.ResponseId);

        Assert.NotNull(selectorMessages);
        Assert.Collection(
            selectorMessages,
            message =>
            {
                Assert.Same(messages[0], message);
                Assert.Equal(ChatRole.System, message.Role);
                Assert.Equal("p0-system", message.Text);
            },
            message =>
            {
                Assert.Same(messages[1], message);
                Assert.Equal(ChatRole.User, message.Role);
                Assert.Equal("p0-user", message.Text);
            });
        Assert.NotNull(downstream.CapturedMessages);
        Assert.Collection(
            downstream.CapturedMessages,
            message =>
            {
                Assert.Same(messages[0], message);
                Assert.Equal(ChatRole.System, message.Role);
                Assert.Equal("p0-system", message.Text);
            },
            message =>
            {
                Assert.Same(messages[1], message);
                Assert.Equal(ChatRole.User, message.Role);
                Assert.Equal("p0-user", message.Text);
            });

        Assert.NotNull(selectorOptions);
        Assert.Equal("p0-downstream", selectorOptions.ModelId);
        Assert.Equal(0.25f, selectorOptions.Temperature);
        Assert.NotNull(downstream.CapturedOptions);
        Assert.Equal("p0-downstream", downstream.CapturedOptions.ModelId);
        Assert.Equal(0.25f, downstream.CapturedOptions.Temperature);
        Assert.Equal(originalModelId, options.ModelId);
        Assert.Equal(originalTemperature, options.Temperature);

        // RoutingContext shallow-clones caller options and forwards that clone.
        Assert.NotSame(options, selectorOptions);
        Assert.NotSame(options, downstream.CapturedOptions);
        Assert.Same(selectorOptions, downstream.CapturedOptions);
        Assert.Equal(cancellationToken, selectorCancellationToken);
        Assert.Equal(cancellationToken, downstream.CapturedCancellationToken);
    }

    private sealed class RecordingChatClient(ChatResponse response) : IChatClient
    {
        public int ResponseCallCount { get; private set; }

        public ChatMessage[]? CapturedMessages { get; private set; }

        public ChatOptions? CapturedOptions { get; private set; }

        public CancellationToken CapturedCancellationToken { get; private set; }

        public Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            ResponseCallCount++;
            CapturedMessages = messages.ToArray();
            CapturedOptions = options;
            CapturedCancellationToken = cancellationToken;
            return Task.FromResult(response);
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            ArgumentNullException.ThrowIfNull(serviceType);
            return serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;
        }

        public void Dispose()
        {
        }
    }
}
