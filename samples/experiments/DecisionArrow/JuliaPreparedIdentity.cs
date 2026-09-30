using System.Security.Cryptography;
using System.Text;
using DecisionInference.Julia;

internal static class JuliaPreparedIdentity
{
    internal static string Hash(JuliaInputBatch batch)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(batch.BatchSize);
            writer.Write(batch.SequenceLength);
            writer.Write(batch.MarkerWidth);
            for (int row = 0; row < batch.BatchSize; row++)
            {
                for (int i = 0; i < batch.SequenceLength; i++) writer.Write(batch.InputIds[row, i]);
                for (int i = 0; i < batch.SequenceLength; i++) writer.Write(batch.AttentionMask[row, i]);
                for (int i = 0; i < batch.MarkerWidth; i++) writer.Write(batch.MarkerPositions[row, i]);
                for (int i = 0; i < batch.MarkerWidth; i++) writer.Write(batch.MarkerMask[row, i]);
                writer.Write(batch.QuestionTypes[row]);
            }
        }
        return Convert.ToHexStringLower(SHA256.HashData(stream.GetBuffer().AsSpan(0, checked((int)stream.Length))));
    }
}
