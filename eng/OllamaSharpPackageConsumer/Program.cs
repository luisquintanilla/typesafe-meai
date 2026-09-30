using System.Net;
using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using DecisionInference;
using DecisionInference.OllamaSharp;
using OllamaSharp;
using OllamaSharp.Models;
using NativeRequest = OllamaSharp.Models.SystemOneRequest;

if (args is ["api", var directory, var destination])
{
    SnapshotApis(directory, destination);
    return;
}
if (args.Length > 0) throw new ArgumentException("Offline consumer accepts no arguments, or api <assembly-directory> <output-json>.");

using var handler = new FixtureHandler();
using var http = new HttpClient(handler) { BaseAddress = new("http://127.0.0.1:11434") };
using var sdk = new OllamaApiClient(http, "captured-default");
IOllamaApiClient client = sdk;
using IDecisionGenerator generator = new OllamaSharpDecisionGenerator(client);
var input = new DecisionInput(JsonSerializer.SerializeToElement("I cannot sign in after resetting my password."),
[
    new BinaryDecisionQuestion("account-access", "Account access?", "Access problem", "No access problem"),
    new ChoiceDecisionQuestion("support-team", "Choose team", [new("account", "Account"), new("billing", "Billing"), new("technical", "Technical")]),
    new ScoreDecisionQuestion("urgency", "Rate urgency", ["Routine", "Blocked", "Critical"])
]);
sdk.SelectedModel = "changed-sdk-default";
var options = new OllamaSharpGenerationOptions { ModelId = "per-call", KeepAliveSeconds = -1 };
var result = await generator.GenerateAsync(input, options.Clone());
var repeated = await generator.GenerateAsync(input);
Require(handler.Models.SequenceEqual(["per-call", "captured-default"]), "Model snapshot or nonsticky override failed.");
Require(handler.Durations.SequenceEqual(["-1s", null]), "Residency mapping failed.");
Require(sdk.SelectedModel == "changed-sdk-default" && options.ModelId == "per-call" && options.KeepAliveSeconds == -1,
    "Caller configuration or options mutated.");
Require(result.ModelId == "per-call" && repeated.ModelId == "captured-default", "Model echo failed.");
Require(result.Answers.Select(a => a.QuestionId).SequenceEqual(["account-access", "support-team", "urgency"]), "Correlation failed.");
Require(result.GetAnswer<BinaryDecisionAnswer>("account-access").TrueProbability == .875, "Binary mapping failed.");
var choice = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
Require(choice.SelectedCandidateId == "technical" && choice.Probabilities.Keys.SequenceEqual(["account", "billing", "technical"]) &&
    choice.Probabilities.Values.SequenceEqual([.125, .125, .75]), "Choice mapping failed.");
var score = result.GetAnswer<ScoreDecisionAnswer>("urgency");
Require(score.Probabilities.SequenceEqual([.25, .25, .5]) && score.NativeScore == 1.25 && score.ExpectedIndex == 1.25, "Score mapping failed.");
Require(result.Usage is { InputTokenCount: int.MaxValue, OutputTokenCount: 1, TotalTokenCount: 2147483648L, ReasoningTokenCount: null },
    "Int64 sum of SDK counters failed.");
Require(result.Answers.All(a => a.Precision == DecisionPrecision.HighPrecision && a.RawRepresentation is JsonElement) &&
    result.RawRepresentation is JsonElement && Equals(result.AdditionalProperties!["ollamasharp.rawRepresentationKind"], "sdk-dto-snapshot"),
    "Owned snapshot evidence failed.");
Require(((JsonElement)choice.RawRepresentation!).GetProperty("confidence").GetDouble() == .33040805446422084 &&
    ((JsonElement)score.RawRepresentation!).GetProperty("confidence").GetDouble() == .053605369642814038,
    "Native entropy concentration snapshot failed.");
Require(ReferenceEquals(generator.GetService(typeof(IOllamaApiClient)), client) &&
    generator.GetService(typeof(Microsoft.Extensions.AI.IChatClient)) is null, "Service isolation failed.");
generator.Dispose();
try
{
    await generator.GenerateAsync(input);
    throw new InvalidOperationException("Disposed generator accepted a call.");
}
catch (ObjectDisposedException) { }
await sdk.SystemOneAsync(new NativeRequest
{
    Model = "borrowed-sdk-still-usable", State = input.State,
    Questions = new()
    {
        ["q0"] = new SystemOneNoulQuestion { Instructions = "Account access?" },
        ["q1"] = new SystemOneChoiceQuestion { Instructions = "Choose team", Criteria = new() { ["0"] = "Account", ["1"] = "Billing", ["2"] = "Technical" } },
        ["q2"] = new SystemOneScoreQuestion { Instructions = "Rate urgency", Criteria = ["Routine", "Blocked", "Critical"] }
    }
});
Require(handler.Models.Count == 3, "Borrowed SDK lifetime or one-call delegation failed.");
Require(((JsonElement)result.RawRepresentation!).GetProperty("model").GetString() == "per-call", "Snapshot did not survive disposal.");
Console.WriteLine("PASS: fresh SDK-only package consumer; stock IOllamaApiClient, all kinds, options/defaults, Int64 usage, borrowed lifetime, owned snapshots; no daemon.");

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static void SnapshotApis(string directory, string destination)
{
    directory = Path.GetFullPath(directory);
    AssemblyLoadContext.Default.Resolving += (_, name) =>
        File.Exists(Path.Combine(directory, name.Name + ".dll"))
            ? AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(directory, name.Name + ".dll")) : null;
    var signatures = new SortedDictionary<string, string[]>(StringComparer.Ordinal);
    foreach (string file in Directory.GetFiles(directory, "DecisionInference.*.dll").Order(StringComparer.Ordinal))
    {
        var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(file);
        var entries = new List<string>();
        foreach (var type in assembly.GetExportedTypes().OrderBy(t => t.FullName, StringComparer.Ordinal))
        {
            entries.Add($"TYPE {Name(type)} attributes={type.Attributes} base={Name(type.BaseType)} interfaces=[{string.Join(",", type.GetInterfaces().Select(Name).Order(StringComparer.Ordinal))}] {Attributes(type.CustomAttributes)}");
            foreach (var member in type.GetMembers(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic))
            {
                string? signature = member switch
                {
                    MethodInfo m when Visible(m) => $"METHOD {m.Attributes} {Name(m.ReturnType)} {m.Name}{Generics(m)}({Parameters(m.GetParameters())}) return:{Attributes(m.ReturnParameter.CustomAttributes)} modifiers:{Modifiers(m.ReturnParameter)}",
                    ConstructorInfo c when Visible(c) => $"CTOR {c.Attributes} {c.Name}({Parameters(c.GetParameters())})",
                    PropertyInfo p when p.GetAccessors(true).Any(Visible) => $"PROPERTY {Name(p.PropertyType)} {p.Name}[{Parameters(p.GetIndexParameters())}] accessors:{string.Join(",", p.GetAccessors(true).Select(a => $"{a.Name}:{a.Attributes}"))} modifiers:{string.Join(",", p.GetRequiredCustomModifiers().Select(Name))}",
                    FieldInfo f when f.IsPublic || f.IsFamily || f.IsFamilyOrAssembly => $"FIELD {f.Attributes} {Name(f.FieldType)} {f.Name} constant:{(f.IsLiteral ? JsonSerializer.Serialize(f.GetRawConstantValue()) : "-")} modifiers:{string.Join(",", f.GetRequiredCustomModifiers().Select(Name))}",
                    EventInfo e when e.AddMethod is { } a && Visible(a) => $"EVENT {Name(e.EventHandlerType)} {e.Name}",
                    _ => null
                };
                if (signature is not null) entries.Add($"{Name(type)} {signature} {Attributes(member.CustomAttributes)}");
            }
        }
        signatures.Add(assembly.GetName().Name!, entries.Order(StringComparer.Ordinal).ToArray());
    }
    if (signatures.Count < 10 || signatures.Values.Sum(v => v.Length) == 0) throw new InvalidOperationException("Incomplete API snapshot.");
    File.WriteAllText(destination, JsonSerializer.Serialize(signatures, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"{signatures.Count} assemblies; {signatures.Values.Sum(v => v.Length)} full API entries: {destination}");
}
static bool Visible(MethodBase m) => m.IsPublic || m.IsFamily || m.IsFamilyOrAssembly;
static string Name(Type? t) => t is null ? "" : t.IsArray ? Name(t.GetElementType()) + "[" + new string(',', t.GetArrayRank() - 1) + "]"
    : t.IsByRef ? Name(t.GetElementType()) + "&" : t.IsPointer ? Name(t.GetElementType()) + "*"
    : t.IsGenericParameter ? t.Name : t.IsGenericType
        ? t.GetGenericTypeDefinition().FullName + "<" + string.Join(",", t.GetGenericArguments().Select(Name)) + ">" : t.FullName ?? t.Name;
static string Attributes(IEnumerable<CustomAttributeData> attributes) => string.Join(" ", attributes.Select(a => a.ToString()).Order(StringComparer.Ordinal));
static string Parameters(ParameterInfo[] parameters) => string.Join(", ", parameters.Select(p =>
    $"{p.Attributes} {Name(p.ParameterType)} {p.Name} default:{(p.HasDefaultValue ? JsonSerializer.Serialize(p.DefaultValue) : "-")} {Modifiers(p)} {Attributes(p.CustomAttributes)}"));
static string Modifiers(ParameterInfo p) => string.Join(",", p.GetRequiredCustomModifiers().Select(Name));
static string Generics(MethodInfo method) => method.IsGenericMethodDefinition ? "<" + string.Join(",", method.GetGenericArguments().Select(t =>
    $"{t.Name}:{t.GenericParameterAttributes}:{string.Join("&", t.GetGenericParameterConstraints().Select(Name))}")) + ">" : "";

internal sealed class FixtureHandler : HttpMessageHandler
{
    internal List<string> Models { get; } = [];
    internal List<string?> Durations { get; } = [];
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        if (request.Method != HttpMethod.Post || request.RequestUri!.AbsolutePath != "/v1/systemone")
            throw new InvalidOperationException("Unexpected endpoint or discovery.");
        using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
        var body = json.RootElement;
        var questions = body.GetProperty("questions");
        if (!questions.EnumerateObject().Select(p => p.Name).SequenceEqual(["q0", "q1", "q2"]) ||
            body.GetProperty("state").ValueKind != JsonValueKind.String ||
            body.GetProperty("state").GetString() != "I cannot sign in after resetting my password." ||
            questions.GetProperty("q1").GetProperty("criteria").GetProperty("2").GetString() != "Technical")
            throw new InvalidOperationException("Native aliases, state or complete schema changed.");
        string model = body.GetProperty("model").GetString()!;
        Models.Add(model);
        Durations.Add(body.TryGetProperty("keep_alive", out var duration) ? duration.GetString() : null);
        string response = """
            {"model":$MODEL,"usage":{"input_tokens":2147483647,"output_tokens":1},
             "answers":{"q0":{"noul":0.875,"type":"noul"},
                        "q1":{"choice":"2","probabilities":{"0":0.125,"1":0.125,"2":0.75},"confidence":0.33040805446422084,"type":"choice"},
                        "q2":{"score":1.25,"probabilities":{"0":0.25,"1":0.25,"2":0.5},
                              "legend":{"0":"Routine","1":"Blocked","2":"Critical"},"confidence":0.053605369642814038,"type":"score"}}}
            """.Replace("$MODEL", JsonSerializer.Serialize(model), StringComparison.Ordinal);
        return new(HttpStatusCode.OK) { Content = new StringContent(response) };
    }
}
