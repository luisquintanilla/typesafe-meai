using System.Security.Cryptography;
using DecisionInference.Arrow;

internal static class JuliaAssetIdentity
{
    internal static readonly IReadOnlyDictionary<string, string> Hashes =
        new System.Collections.ObjectModel.ReadOnlyDictionary<string, string>(new SortedDictionary<string, string>(StringComparer.Ordinal)
        {
            ["model.onnx"] = "97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9",
            ["model.onnx.data"] = "fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1",
            ["tokenizer.json"] = "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"
        });
    internal const string QuestionsHash = "9e6e6c5c580934309cb1f9b4572c7b87a93ce22b32b50cb8b3af20caa20068ea";
    internal const string StatesHash = "c00636dc21f7a4e3bb099c508fbf24a407af546851e060c5852cb9aca54639fa";
    internal const string Preprocessing = "julia-python-json-strict1024-head256-option48-min16-reserved0to4-pad8-v1";
    internal const string Decoder = "julia-valid-fp32-logits-double-stable-full-softmax-firstargmax-binary-p1-zero-based-highprecision-native-null-v1";

    internal static async Task VerifyAsync(string directory, CancellationToken token)
    {
        foreach (var expected in Hashes)
            if (await HashFileAsync(Path.Combine(directory, expected.Key), token) != expected.Value)
                throw new InvalidDataException($"Pinned Julia loader asset changed: {expected.Key}.");
    }

    internal static async Task<string> HashFileAsync(string path, CancellationToken token)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, token));
    }
}
