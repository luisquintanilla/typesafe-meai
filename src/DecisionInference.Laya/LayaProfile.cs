using System.Collections.ObjectModel;
using System.Text.Json;

namespace DecisionInference.Laya;

public sealed class LayaProfile
{
    public const string Revision = "68f27dfe5a27a54fb2b1fefc432f43f972e90868";
    public const string ModelId = "receptron/laya-onnx@" + Revision;
    public int MaxLength { get; }
    public int HeadMaxLength { get; }
    public IReadOnlyDictionary<string, float> Temperatures { get; }

    public LayaProfile(int maxLength, int headMaxLength, IEnumerable<KeyValuePair<string, float>> temperatures)
    {
        ArgumentNullException.ThrowIfNull(temperatures);
        if (maxLength < 4 || maxLength > 512 || headMaxLength < 16 || headMaxLength > maxLength - 4)
            throw new ArgumentException("Invalid English FP32 sequence/head limits.");
        var copy = temperatures.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
        if (copy.Values.Any(t => !float.IsFinite(t) || t <= 0) ||
            new[] { "choice", "score", "noul" }.Any(k => !copy.ContainsKey(k)))
            throw new ArgumentException("Finite positive base temperatures for all three kinds are required.");
        MaxLength = maxLength;
        HeadMaxLength = headMaxLength;
        Temperatures = new ReadOnlyDictionary<string, float>(copy);
    }

    public static LayaProfile Load(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
        var root = doc.RootElement;
        var bases = root.GetProperty("temperature").EnumerateArray().Select(t => t.GetSingle()).ToArray();
        if (bases.Length != 3) throw new InvalidDataException("Expected Choice, Score, Noul temperatures.");
        Dictionary<string, float> values = new(StringComparer.Ordinal)
        {
            ["choice"] = bases[0], ["score"] = bases[1], ["noul"] = bases[2]
        };
        foreach (var p in root.GetProperty("temperature_by_options").EnumerateObject()) values.Add(p.Name, p.Value.GetSingle());
        return new(root.GetProperty("max_len").GetInt32(), root.GetProperty("head_max_len").GetInt32(), values);
    }

    /// <summary>Source-profile policy clamps temperature to [0.5,5]; original values remain in Temperatures.</summary>
    public float Temperature(DecisionKind kind, int count)
    {
        string name = KindName(kind);
        string bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+";
        return Math.Clamp(Temperatures.TryGetValue($"{name}:{bucket}", out var t) ? t : Temperatures[name], 0.5f, 5f);
    }

    internal static string KindName(DecisionKind kind) => kind switch
    {
        DecisionKind.Binary => "noul", DecisionKind.Choice => "choice", DecisionKind.Score => "score",
        _ => throw new ArgumentOutOfRangeException(nameof(kind))
    };
}
