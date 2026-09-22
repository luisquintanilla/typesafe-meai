using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>Pre-cache native normalization shared with direct provider entry points.</summary>
public static class TypeSafeChatClientBuilderExtensions
{
    /// <summary>Place first (outermost): validates even JSON-ignored options before cache lookup.</summary>
    public static ChatClientBuilder UseTypeSafeRequestNormalization(this ChatClientBuilder builder, TypeSafeClientOptions options)
    {
        options.Validate();
        return builder.Use(inner => new NormalizingClient(inner, options));
    }

    private sealed class NormalizingClient(IChatClient inner, TypeSafeClientOptions settings) : DelegatingChatClient(inner)
    {
        public override Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var request = NativeRequest.Create(messages, options, settings);
            return InnerClient.GetResponseAsync([request.State.ToMessage()], request.Options, cancellationToken);
        }

        public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
            ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var request = NativeRequest.Create(messages, options, settings);
            await foreach (var update in InnerClient.GetStreamingResponseAsync([request.State.ToMessage()], request.Options, cancellationToken).ConfigureAwait(false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return update;
            }
        }
    }
}
