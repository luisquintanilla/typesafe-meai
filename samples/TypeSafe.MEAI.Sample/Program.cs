using Microsoft.Extensions.AI;
using TypeSafe.MEAI;
using TypeSafe.MEAI.Sample;

Environment.SetEnvironmentVariable("DOTNET_AIEVAL_TELEMETRY_OPTOUT", "1");
string scenario = args.FirstOrDefault() ?? "native";
using var environment = new SampleEnvironment(useCache: scenario is "pipeline" or "schema");
IChatClient client = environment.Client;
switch (scenario)
{
    case "native":
    case "pipeline":
        var answer = await client.GetTypeSafeResponseAsync(SampleEnvironment.State, SampleEnvironment.Questions);
        Console.WriteLine(answer.Json);
        Console.WriteLine($"Noul={answer.GetAnswer<NoulAnswer>("followup").Noul}, Choice={answer.GetAnswer<ChoiceAnswer>("destination").Choice}, fractional Score={answer.GetAnswer<ScoreAnswer>("quality").Score}");
        if (scenario == "pipeline")
        {
            var cached = await client.GetTypeSafeResponseAsync(SampleEnvironment.State, SampleEnvironment.Questions);
            if (environment.Handler.Sends != 1 || cached.Json.GetRawText() != answer.Json.GetRawText())
                throw new InvalidOperationException("Native cache proof failed.");
            Console.WriteLine($"Two calls; {environment.Handler.Sends} HTTP send. Cached usage is original usage, not new billable usage.");
        }
        Console.WriteLine($"Actual model={answer.Model}, input={answer.InputTokens}, output={answer.OutputTokens}, total={answer.TotalTokens}.");
        break;
    case "evaluate":
        await IntegrationScenarios.EvaluateAsync(environment, args.ElementAtOrDefault(1) ?? Path.Combine("artifacts", "reports"));
        break;
    case "tool":
        await IntegrationScenarios.ToolAsync(environment);
        break;
    case "schema":
        await SchemaScenario.RunAsync(environment);
        break;
    case "route":
        await CompositionScenarios.RouteAsync(environment);
        break;
    case "select-tool":
    case "dispatch":
        await CompositionScenarios.SelectToolAsync(environment, dispatch: scenario == "dispatch");
        break;
    default:
        throw new ArgumentException("Supported offline scenarios: native, pipeline, evaluate, tool, schema, route, select-tool, dispatch.");
}
