namespace DecisionInference;

/// <summary>Exact question-ID access to the response's ordered answers.</summary>
public static class DecisionResponseExtensions
{
    /// <summary>Gets an answer by its unmodified ordinal ID; throws <see cref="KeyNotFoundException"/> when absent.</summary>
    public static DecisionAnswer GetAnswer(this DecisionResponse response, string questionId)
    {
        ArgumentNullException.ThrowIfNull(response);
        ArgumentException.ThrowIfNullOrWhiteSpace(questionId);
        return response.Answers.FirstOrDefault(a => string.Equals(a.QuestionId, questionId, StringComparison.Ordinal)) ??
            throw new KeyNotFoundException($"No answer exists for question ID '{questionId}'.");
    }

    /// <summary>Gets an exact-ID answer; throws <see cref="InvalidOperationException"/> if its type differs.</summary>
    public static TAnswer GetAnswer<TAnswer>(this DecisionResponse response, string questionId) where TAnswer : DecisionAnswer =>
        response.GetAnswer(questionId) as TAnswer ??
            throw new InvalidOperationException($"Answer '{questionId}' is not a {typeof(TAnswer).Name}.");
}
