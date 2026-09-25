using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Http.Resilience;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Polly;

namespace TypeSafe.MEAI;

/// <summary>Keyed provider registration with one HTTP retry owner and safe MEAI middleware.</summary>
public static class TypeSafeServiceCollectionExtensions
{
    /// <summary>Registers a keyed decorated client; caching is explicitly opt-in. Returns its HTTP builder for handler injection.</summary>
    public static IHttpClientBuilder AddTypeSafeChatClient(this IServiceCollection services, string key, string apiKey,
        TypeSafeClientOptions options, bool useCache = false, TimeSpan? attemptTimeout = null, TimeSpan? totalTimeout = null)
    {
        options = options.Validate();
        if (useCache) ArgumentException.ThrowIfNullOrWhiteSpace(options.CacheNamespace);
        var name = $"typesafe:{key}";
        var http = services.AddHttpClient(name, client => client.Timeout = Timeout.InfiniteTimeSpan)
            .ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler { AllowAutoRedirect = false })
            .RemoveAllLoggers();
        http.AddResilienceHandler("native", (pipeline, context) =>
        {
            pipeline.TimeProvider = context.ServiceProvider.GetService<TimeProvider>() ?? TimeProvider.System;
            pipeline.AddTimeout(totalTimeout ?? TimeSpan.FromSeconds(30));
            pipeline.AddRetry(new HttpRetryStrategyOptions
            {
                MaxRetryAttempts = 2, Delay = TimeSpan.FromMilliseconds(500), MaxDelay = TimeSpan.FromSeconds(5),
                BackoffType = DelayBackoffType.Exponential, UseJitter = true, ShouldRetryAfterHeader = false,
                DelayGenerator = args =>
                {
                    var header = args.Outcome.Result?.Headers.RetryAfter;
                    TimeSpan? delay = header?.Delta;
                    if (header?.Date is { } date)
                        delay = date - pipeline.TimeProvider.GetUtcNow();
                    return new ValueTask<TimeSpan?>(delay is { } value && value < TimeSpan.Zero ? TimeSpan.Zero : delay);
                },
                OnRetry = args =>
                {
                    args.Outcome.Result?.Dispose();
                    return default;
                }
            });
            pipeline.AddTimeout(attemptTimeout ?? TimeSpan.FromSeconds(10));
        });
        http.AddHttpMessageHandler(() => new BoundedResponseHandler(options.MaximumResponseBytes));
        services.AddKeyedChatClient(key, sp =>
        {
            var raw = new NativeChatClient(sp.GetRequiredService<IHttpClientFactory>().CreateClient(name), options, apiKey, ownsHttpClient: true);
            var builder = raw.AsBuilder().UseTypeSafeRequestNormalization(options)
                .Use(inner => new LoggingChatClient(inner, new NoTraceLogger((sp.GetService<ILoggerFactory>() ?? NullLoggerFactory.Instance).CreateLogger("TypeSafe.MEAI"))))
                .UseOpenTelemetry(sourceName: "TypeSafe.MEAI", configure: telemetry => telemetry.EnableSensitiveData = false);
            if (useCache) builder.UseDistributedCache(sp.GetRequiredService<IDistributedCache>(),
                configure: cache => cache.JsonSerializerOptions = TypeSafeChatClient.ContentJson);
            return new TypeSafeChatClient(builder.Build(sp), options);
        });
        return http;
    }

    private sealed class NoTraceLogger(ILogger inner) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => inner.BeginScope(state);
        public bool IsEnabled(LogLevel logLevel) => logLevel != LogLevel.Trace && inner.IsEnabled(logLevel);
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            if (IsEnabled(logLevel)) inner.Log(logLevel, eventId, state, exception, formatter);
        }
    }
}
