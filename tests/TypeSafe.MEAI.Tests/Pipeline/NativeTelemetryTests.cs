using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Net;
using System.Text;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Pipeline;

[CollectionDefinition("Stage A telemetry environment", DisableParallelization = true)]
public sealed class TelemetryEnvironmentCollection;

[Trait("Stage", "A")]
[Collection("Stage A telemetry environment")]
public sealed class NativeTelemetryTests
{
    [Fact]
    public async Task DefaultTelemetry_RedactsSensitiveContentEvenWithOptInEnvironmentAndAllLogLevels()
    {
        const string variable = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";
        var original = Environment.GetEnvironmentVariable(variable);
        try
        {
            Environment.SetEnvironmentVariable(variable, "true");
            using var telemetry = new TelemetryCapture();
            var logs = new RecordingLogs();
            using var handler = new RecordingHandler((_, attempt, _) => Task.FromResult(attempt switch
            {
                1 => NativeFixture.Response(NativeFixture.Envelope.Replace("noul-extra", "SYNTHETIC-ANSWER-MARKER", StringComparison.Ordinal)),
                2 => NativeFixture.Response("SYNTHETIC-HTTP-PRIVATE-BODY", 422),
                _ => NativeFixture.Response("SYNTHETIC-PROTOCOL-PRIVATE-BODY")
            }));
            var services = new ServiceCollection();
            services.AddLogging(builder => builder.SetMinimumLevel(LogLevel.Trace).AddProvider(logs));
            services.AddDistributedMemoryCache();
            services.AddTypeSafeChatClient("native", NativeFixture.Key, NativeFixture.Settings, useCache: true)
                .ConfigurePrimaryHttpMessageHandler(() => handler);
            using var provider = services.BuildServiceProvider();
            var client = provider.Client();
            var questions = NativeFixture.Questions();
            questions[0] = new NoulQuestion("follow-up", TypeSafeJson.Entry("SYNTHETIC-QUESTION-MARKER"));
            var options = NativeFixture.Options(questions);
            ChatMessage[] Messages(string suffix) => [new(ChatRole.User, "SYNTHETIC-STATE-MARKER" + suffix)];
            var first = await client.GetResponseAsync(Messages("success"), options, TestContext.Current.CancellationToken);
            var cached = await client.GetResponseAsync(Messages("success"), options, TestContext.Current.CancellationToken);
            Assert.Equal(0.82, Assert.IsType<NoulAnswer>(Assert.Single(first.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Answers["follow-up"]).Noul);
            Assert.Equal(0.91, Assert.IsType<ChoiceAnswer>(Assert.Single(cached.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Answers["destination"]).Confidence);
            Assert.Null(cached.RawRepresentation);
            Assert.Equal(1, handler.Sends);
            var httpError = await Assert.ThrowsAsync<TypeSafeApiException>(() => client.GetResponseAsync(Messages("http"), options, TestContext.Current.CancellationToken));
            Assert.Equal(HttpStatusCode.UnprocessableEntity, httpError.StatusCode);
            await Assert.ThrowsAsync<TypeSafeProtocolException>(() => client.GetResponseAsync(Messages("protocol"), options, TestContext.Current.CancellationToken));
            Assert.Equal(3, handler.Sends);
            Assert.NotEmpty(logs.Entries);
            Assert.NotEmpty(telemetry.Activities);
            Assert.NotEmpty(telemetry.Measurements);
            var all = string.Join("\n", logs.Entries.Concat(telemetry.Activities.Select(a => a.Serialized))
                .Concat(telemetry.Measurements.Select(m => m.Serialized)));
            string[] secrets =
            [
                NativeFixture.Key, "SYNTHETIC-STATE-MARKER", "SYNTHETIC-QUESTION-MARKER", "SYNTHETIC-ANSWER-MARKER",
                "SYNTHETIC-HTTP-PRIVATE-BODY", "SYNTHETIC-PROTOCOL-PRIVATE-BODY", "billing", "technical",
                "0.82", "0.91", "0.76", "1.25", "Authorization", "Bearer"
            ];
            Assert.All(secrets, secret => Assert.DoesNotContain(secret, all, StringComparison.OrdinalIgnoreCase));
            Assert.Equal(4, telemetry.Activities.Count);
            Assert.All(telemetry.Activities, a =>
            {
                Assert.Equal("chat", a.Tags["gen_ai.operation.name"]);
                Assert.Equal("jev-1.13.0", a.Tags["gen_ai.request.model"]);
            });
            Assert.Equal(2, telemetry.Activities.Count(a => a.Status == ActivityStatusCode.Error));
            Assert.Contains(logs.Entries, entry => entry.Contains("TypeSafe", StringComparison.Ordinal));
        }
        finally
        {
            Environment.SetEnvironmentVariable(variable, original);
        }
    }

    [Fact]
    public async Task Telemetry_SeparatesTwoCachedResponseUsagesFromOneNativeSend()
    {
        using var telemetry = new TelemetryCapture();
        using var handler = new RecordingHandler();
        using var provider = NativeFixture.Services(handler, cache: true);
        var first = await provider.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        var cached = await provider.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        NativeFixture.AssertChat(first);
        NativeFixture.AssertChat(cached);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(2, telemetry.Activities.Count);
        var tokens = telemetry.Measurements.Where(m => m.Name == "gen_ai.client.token.usage").ToArray();
        Assert.Equal([11d, 11d], tokens.Where(m => Equals(m.Tags["gen_ai.token.type"], "input")).Select(m => m.Value));
        Assert.Equal([7d, 7d], tokens.Where(m => Equals(m.Tags["gen_ai.token.type"], "output")).Select(m => m.Value));
        Assert.Equal(2, telemetry.Measurements.Count(m => m.Name == "gen_ai.client.operation.duration"));
        Assert.All(telemetry.Activities, a => Assert.Equal("jev-1.13.0", a.Tags["gen_ai.response.model"]));
    }

    [Fact]
    public async Task BufferedStreamingTelemetry_EmitsCompletionOnlyAfterFullBodyRead()
    {
        using var telemetry = new TelemetryCapture();
        var stream = new TrackedStream(Encoding.UTF8.GetBytes(NativeFixture.Envelope), stall: true);
        using var handler = new RecordingHandler((_, _, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new TrackedContent(stream) }));
        using var services = NativeFixture.Services(handler);
        var pending = services.Client().GetStreamingResponseAsync(NativeFixture.Messages(), NativeFixture.Options(),
            TestContext.Current.CancellationToken).ToChatResponseAsync(TestContext.Current.CancellationToken);
        await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.False(pending.IsCompleted);
        Assert.Empty(telemetry.Activities);
        Assert.Empty(telemetry.Measurements);
        stream.Release.SetResult();
        NativeFixture.AssertChat(await pending);
        Assert.Single(telemetry.Activities);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(2, telemetry.Measurements.Count(m => m.Name == "gen_ai.client.token.usage"));
        Assert.True(stream.Disposed);
    }

    private sealed class RecordingLogs : ILoggerProvider
    {
        internal ConcurrentQueue<string> Entries { get; } = new();
        public ILogger CreateLogger(string categoryName) => new Logger(categoryName, Entries);
        public void Dispose() { }
        private sealed class Logger(string category, ConcurrentQueue<string> entries) : ILogger
        {
            public bool IsEnabled(LogLevel logLevel) => true;
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull
            {
                entries.Enqueue($"scope:{state}");
                return null;
            }
            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            {
                var fields = state is IEnumerable<KeyValuePair<string, object?>> properties
                    ? string.Join(";", properties.Select(p => $"{p.Key}={p.Value}")) : state?.ToString();
                entries.Enqueue($"{category}|{logLevel}|{formatter(state, exception)}|{fields}|{exception}");
            }
        }
    }

    private sealed record CapturedActivity(Dictionary<string, object?> Tags, ActivityStatusCode Status, string Serialized);
    private sealed record CapturedMeasurement(string Name, double Value, Dictionary<string, object?> Tags, string Serialized);

    private sealed class TelemetryCapture : IDisposable
    {
        private readonly ActivityListener _activities;
        private readonly MeterListener _metrics;
        internal ConcurrentQueue<CapturedActivity> Activities { get; } = new();
        internal ConcurrentQueue<CapturedMeasurement> Measurements { get; } = new();
        internal TelemetryCapture()
        {
            _activities = new()
            {
                ShouldListenTo = source => source.Name == "TypeSafe.MEAI",
                Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
                SampleUsingParentId = (ref ActivityCreationOptions<string> _) => ActivitySamplingResult.AllDataAndRecorded,
                ActivityStopped = activity =>
                {
                    var tags = activity.TagObjects.ToDictionary();
                    var values = tags.Select(p => $"{p.Key}={p.Value}")
                        .Concat(activity.Baggage.Select(p => $"{p.Key}={p.Value}"))
                        .Concat(activity.Events.Select(e => $"{e.Name}:{string.Join(";", e.Tags.Select(t => $"{t.Key}={t.Value}"))}"))
                        .Concat(activity.Links.Select(l => string.Join(";", l.Tags!.Select(t => $"{t.Key}={t.Value}"))));
                    Activities.Enqueue(new(tags, activity.Status, $"{activity.DisplayName}|{activity.StatusDescription}|{string.Join("|", values)}"));
                }
            };
            ActivitySource.AddActivityListener(_activities);
            _metrics = new()
            {
                InstrumentPublished = (instrument, listener) =>
                {
                    if (instrument.Meter.Name == "TypeSafe.MEAI")
                        listener.EnableMeasurementEvents(instrument);
                }
            };
            _metrics.SetMeasurementEventCallback<int>((instrument, value, tags, _) => Record(instrument, value, tags));
            _metrics.SetMeasurementEventCallback<long>((instrument, value, tags, _) => Record(instrument, value, tags));
            _metrics.SetMeasurementEventCallback<double>((instrument, value, tags, _) => Record(instrument, value, tags));
            _metrics.Start();
        }
        private void Record(Instrument instrument, double value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            var values = tags.ToArray().ToDictionary();
            Measurements.Enqueue(new(instrument.Name, value, values,
                $"{instrument.Name}={value}|{string.Join("|", values.Select(p => $"{p.Key}={p.Value}"))}"));
        }
        public void Dispose()
        {
            _activities.Dispose();
            _metrics.Dispose();
        }
    }
}
