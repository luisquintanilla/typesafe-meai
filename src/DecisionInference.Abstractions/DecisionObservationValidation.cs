namespace DecisionInference;

internal static class DecisionObservationValidation
{
    internal readonly record struct ScoreObservation(double Expected, double Lower, double Upper);

    internal static void Unit(double value)
    {
        if (!double.IsFinite(value) || value is < 0 or > 1)
            throw new DecisionProtocolException("Probability must be finite and in [0,1].");
    }

    internal static void Binary(double value, DecisionPrecision precision)
    {
        Unit(value);
        if (precision != DecisionPrecision.HighPrecision) DecisionRounding.Interval(value, 1, precision);
    }

    internal static void Distribution(ReadOnlySpan<double> values, DecisionPrecision precision)
    {
        double sum = 0;
        foreach (double value in values) { Unit(value); sum += value; }
        if (precision == DecisionPrecision.HighPrecision)
        {
            if (Math.Abs(sum - 1) > DecisionAnswer.DistributionTolerance)
                throw new DecisionProtocolException("Distribution sum outside absolute tolerance 1e-6.");
            return;
        }
        decimal lower = 0, upper = 0;
        foreach (double value in values)
        {
            var interval = DecisionRounding.Interval(value, 1, precision);
            lower += interval.Lower;
            upper += interval.Upper;
        }
        if (lower > 1 || upper < 1)
            throw new DecisionProtocolException("Rounded observations cannot represent a normalized distribution.");
    }

    internal static ScoreObservation Score(ReadOnlySpan<double> values, double? nativeScore, DecisionPrecision precision)
    {
        if (values.Length < 2) throw new DecisionProtocolException("An ordinal distribution requires at least two levels.");
        Distribution(values, precision);
        double expected = 0;
        for (int i = 0; i < values.Length; i++) expected += i * values[i];
        double tolerance = DecisionAnswer.DistributionTolerance * (values.Length - 1);
        double lower = Math.Max(0, expected - tolerance), upper = Math.Min(values.Length - 1, expected + tolerance);
        if (precision != DecisionPrecision.HighPrecision)
        {
            var bounds = DecisionRounding.ExpectedBounds(values, precision);
            lower = (double)bounds.Lower;
            upper = (double)bounds.Upper;
            if (nativeScore is { } rounded)
            {
                var interval = DecisionRounding.Interval(rounded, values.Length - 1, precision);
                if (interval.Upper < bounds.Lower || interval.Lower > bounds.Upper)
                    throw new DecisionProtocolException("Native score is incompatible with rounded probability observations.");
            }
        }
        else if (nativeScore is { } n && (!double.IsFinite(n) || n < 0 || n > values.Length - 1 ||
            Math.Abs(n - expected) > tolerance))
            throw new DecisionProtocolException("Native score disagrees with the distribution's expected index.");
        return new(expected, lower, upper);
    }
}
