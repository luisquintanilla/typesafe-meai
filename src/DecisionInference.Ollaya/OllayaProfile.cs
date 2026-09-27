using System.Text.Json;

namespace DecisionInference.Ollaya;

// Narrow audited server layouts, not a model registry. The pinned server overstates guard capabilities.
internal sealed record OllayaProfile(string Family, string Layout, string Capabilities)
{
    internal static OllayaProfile Read(JsonElement show)
    {
        try
        {
            if (show.GetProperty("router").ValueKind != JsonValueKind.Null)
                throw new NotSupportedException("Configure a concrete Ollaya model; router semantics cannot be established by this adapter.");
            var details = show.GetProperty("details");
            string family = details.GetProperty("family").GetString()!;
            var info = show.GetProperty("model_info");
            if (!info.TryGetProperty(family + ".layout", out var layoutValue))
                throw new NotSupportedException("Ollaya did not report an audited arbitrary-question layout.");
            string? layout = layoutValue.GetString();
            if ((family, layout) is not (("nli", "nli-pairs-v1") or ("gliclass", "gliclass-uni-v1")))
                throw new NotSupportedException("This Ollaya layout has not established nontruncating arbitrary-question semantics. Guards and silent Laya head truncation are unsupported; use embedded Laya for strict local preflight.");
            if (info.GetProperty("general.architecture").GetString() != family)
                throw new DecisionProtocolException("Ollaya family and architecture disagree.");
            if (details.GetProperty("format").GetString() != "onnx" || info.GetProperty("general.engine").GetString() != "onnxruntime")
                throw new NotSupportedException("This adapter currently supports audited ONNX server layouts only.");
            var capabilities = show.GetProperty("capabilities").EnumerateArray().Select(v =>
                v.ValueKind == JsonValueKind.String ? v.GetString()! :
                throw new DecisionProtocolException("Ollaya capabilities must contain only strings.")).ToArray();
            return new(family, layout!, string.Join(",", capabilities.Order(StringComparer.Ordinal)));
        }
        catch (Exception error) when (error is InvalidOperationException or KeyNotFoundException or ArgumentException)
        {
            throw new DecisionProtocolException("Malformed Ollaya model discovery.");
        }
    }

    internal void Validate(DecisionRequest request)
    {
        var kinds = Capabilities.Split(',').ToHashSet(StringComparer.Ordinal);
        if (Family == "gliclass" && request.State.ValueKind == JsonValueKind.String && request.State.GetString()!.Length == 0)
            throw new NotSupportedException("GLiClass cannot report instruction truncation reliably for empty string state.");
        foreach (var q in request.Questions)
        {
            string kind = q.Kind switch { DecisionKind.Binary => "noul", DecisionKind.Choice => "choice", _ => "score" };
            if (!kinds.Contains(kind)) throw new NotSupportedException("The selected Ollaya model does not report this decision kind.");
            if (Family is "nli" or "gliclass" && q is BinaryDecisionQuestion binary &&
                (!string.IsNullOrWhiteSpace(binary.TrueDescription) != !string.IsNullOrWhiteSpace(binary.FalseDescription)))
                throw new NotSupportedException("This Ollaya layout would ignore a one-sided Binary description; supply both or neither.");
        }
        string[] markers = Family switch
        {
            "gliclass" => ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"],
            _ => ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<|endoftext|>"]
        };
        void Check(string text)
        {
            if (markers.Any(m => text.Contains(m, StringComparison.Ordinal)))
                throw new NotSupportedException("The selected Ollaya layout would reinterpret or rewrite reserved marker input.");
        }
        void CheckJson(JsonElement value)
        {
            if (value.ValueKind == JsonValueKind.String) Check(value.GetString()!);
            else if (value.ValueKind == JsonValueKind.Object)
                foreach (var property in value.EnumerateObject()) { Check(property.Name); CheckJson(property.Value); }
            else if (value.ValueKind == JsonValueKind.Array)
                foreach (var item in value.EnumerateArray()) CheckJson(item);
        }
        CheckJson(request.State);
        foreach (var q in request.Questions)
        {
            Check(q.Instructions);
            if (q is ChoiceDecisionQuestion choice)
                foreach (var candidate in choice.Candidates) Check(candidate.Description);
            else if (q is ScoreDecisionQuestion score)
                foreach (var level in score.Rubric) Check(level);
            else if (q is BinaryDecisionQuestion binary)
            {
                if (binary.TrueDescription is not null) Check(binary.TrueDescription);
                if (binary.FalseDescription is not null) Check(binary.FalseDescription);
            }
            if (Family == "gliclass" && request.State.ValueKind == JsonValueKind.String)
                Check(q.Instructions + request.State.GetString());
        }
    }

    internal DecisionRequest Prepare(DecisionRequest request)
    {
        if (Family != "nli") return request;
        // NLI templates can ignore instructions for described outcomes. Retain both semantic inputs.
        string Describe(DecisionQuestion q, string description) =>
            q.Instructions.Length == 0 ? description : q.Instructions + "\n" + description;
        return new(request.State, request.Questions.Select<DecisionQuestion, DecisionQuestion>(q => q switch
        {
            ChoiceDecisionQuestion c => new ChoiceDecisionQuestion(c.Id, c.Instructions,
                c.Candidates.Select(p => new DecisionCandidate(p.Id, Describe(c, p.Description)))),
            ScoreDecisionQuestion s => new ScoreDecisionQuestion(s.Id, s.Instructions, s.Rubric.Select(r => Describe(s, r))),
            BinaryDecisionQuestion b when b.TrueDescription is not null && b.FalseDescription is not null =>
                new BinaryDecisionQuestion(b.Id, b.Instructions, Describe(b, b.TrueDescription), Describe(b, b.FalseDescription)),
            _ => q
        }));
    }
}
