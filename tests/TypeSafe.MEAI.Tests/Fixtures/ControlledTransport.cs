using System.Threading.Channels;
using Microsoft.Extensions.Time.Testing;
using Xunit;

namespace TypeSafe.MEAI.Tests.Fixtures;

internal sealed class TestClock : TimeProvider
{
    private readonly FakeTimeProvider _time = new(new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero));
    private readonly Channel<ScheduledTimer> _scheduled = Channel.CreateUnbounded<ScheduledTimer>();
    public override DateTimeOffset GetUtcNow() => _time.GetUtcNow();
    public override long GetTimestamp() => _time.GetTimestamp();
    public override long TimestampFrequency => _time.TimestampFrequency;
    public override TimeZoneInfo LocalTimeZone => _time.LocalTimeZone;
    internal void Advance(TimeSpan amount) => _time.Advance(amount);
    public override ITimer CreateTimer(TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
    {
        var timer = new ObservedTimer(this, _time.CreateTimer(callback, state, dueTime, period));
        Notify(dueTime);
        return timer;
    }

    private void Notify(TimeSpan dueTime)
    {
        if (dueTime != Timeout.InfiniteTimeSpan)
            _scheduled.Writer.TryWrite(new(dueTime, GetUtcNow() + dueTime));
    }

    internal async Task<ScheduledTimer> NextTimerAsync(Func<TimeSpan, bool> matches)
    {
        using var guard = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        guard.CancelAfter(TimeSpan.FromSeconds(10));
        while (true)
        {
            var timer = await _scheduled.Reader.ReadAsync(guard.Token);
            if (matches(timer.Delay)) return timer;
        }
    }

    internal async Task AdvanceRetryAsync()
    {
        var timer = await NextTimerAsync(delay => delay > TimeSpan.Zero && delay <= TimeSpan.FromSeconds(5));
        Assert.InRange(timer.Delay, TimeSpan.FromTicks(1), TimeSpan.FromSeconds(5));
        Advance(timer.DueAt - GetUtcNow());
    }

    private sealed class ObservedTimer(TestClock owner, ITimer inner) : ITimer
    {
        public bool Change(TimeSpan dueTime, TimeSpan period)
        {
            var changed = inner.Change(dueTime, period);
            if (changed) owner.Notify(dueTime);
            return changed;
        }
        public void Dispose() => inner.Dispose();
        public ValueTask DisposeAsync() => inner.DisposeAsync();
    }
}

internal sealed record ScheduledTimer(TimeSpan Delay, DateTimeOffset DueAt);

internal sealed class TrackedStream(byte[] bytes, bool knownLength = false, bool stall = false, bool failRead = false) : Stream
{
    private readonly MemoryStream _inner = new(bytes, writable: false);
    internal TaskCompletionSource Entered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
    internal TaskCompletionSource Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
    internal TaskCompletionSource Canceled { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
    internal bool Disposed { get; private set; }
    internal int ReadCount { get; private set; }
    public override bool CanRead => true;
    public override bool CanSeek => knownLength;
    public override bool CanWrite => false;
    public override long Length => _inner.Length;
    public override long Position { get => _inner.Position; set => _inner.Position = value; }
    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        ReadCount++;
        Entered.TrySetResult();
        if (stall)
        {
            try { await Release.Task.WaitAsync(cancellationToken); }
            catch (OperationCanceledException) { Canceled.TrySetResult(); throw; }
        }
        if (failRead) throw new IOException("SYNTHETIC-PRIVATE-READ-ERROR");
        return await _inner.ReadAsync(buffer, cancellationToken);
    }
    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
    public override int Read(byte[] buffer, int offset, int count) => throw new InvalidOperationException("Expected asynchronous buffering.");
    public override long Seek(long offset, SeekOrigin origin) => _inner.Seek(offset, origin);
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override void Flush() => throw new NotSupportedException();
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            Disposed = true;
            _inner.Dispose();
        }
        base.Dispose(disposing);
    }
}

internal sealed class TrackedContent(TrackedStream stream) : StreamContent(stream)
{
    internal bool Disposed { get; private set; }
    protected override void Dispose(bool disposing)
    {
        if (disposing) Disposed = true;
        base.Dispose(disposing);
    }
}

internal sealed class TrackedResponse(System.Net.HttpStatusCode status) : HttpResponseMessage(status)
{
    internal bool Disposed { get; private set; }
    protected override void Dispose(bool disposing)
    {
        if (disposing) Disposed = true;
        base.Dispose(disposing);
    }
}
