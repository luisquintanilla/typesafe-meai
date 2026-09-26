using System.Text;

namespace DecisionInference.Tests;

/// <summary>
/// Tiny test-only ONNX protobuf writer (IR8/opset13); no generator/model download dependency.
/// The default graph casts marker positions to logits and scales the first two marker masks
/// to action probabilities. It is an execution/lifetime fixture, NOT a trained model.
/// </summary>
internal static class SyntheticOnnx
{
    internal sealed record Input(string Name, int ElementType, long[] Shape);
    internal static Input[] Inputs() =>
    [
        new("input_ids", 7, [-1, -1]), new("attention_mask", 7, [-1, -1]),
        new("marker_pos", 7, [-1, -1]), new("marker_mask", 9, [-1, -1]),
        new("qtype", 7, [-1])
    ];

    internal static byte[] Create(Input[]? inputs = null, string mode = "dynamic")
    {
        var graph = new Proto();
        graph.Text(2, "synthetic-decision-fixture");
        foreach (var input in inputs ?? Inputs())
            graph.Message(11, Value(input.Name, input.ElementType, input.Shape));
        if (mode == "constant")
        {
            graph.Message(5, FloatTensor("logits_value", [1, 2], [0, 1]));
            graph.Message(5, FloatTensor("action_value", [1, 2], [.5f, .5f]));
            graph.Message(1, Node("Identity", ["logits_value"], "logits"));
            graph.Message(1, Node("Identity", ["action_value"], "act_probs"));
        }
        else
        {
            graph.Message(1, Node("Cast", ["marker_pos"], mode == "transpose-logits" ? "logits_raw" : "logits", IntAttribute("to", 1)));
            if (mode == "transpose-logits")
                graph.Message(1, Node("Transpose", ["logits_raw"], "logits", IntsAttribute("perm", [1, 0])));
            graph.Message(1, Node("Cast", ["marker_mask"], "mask_float", IntAttribute("to", 1)));
            graph.Message(5, IntTensor("starts", [0]));
            graph.Message(5, IntTensor("ends", [2]));
            graph.Message(5, IntTensor("axes", [1]));
            graph.Message(1, Node("Slice", ["mask_float", "starts", "ends", "axes"], "first_two"));
            graph.Message(5, FloatTensor("half", [], [.5f]));
            graph.Message(1, Node("Mul", ["first_two", "half"], mode == "transpose-action" ? "action_raw" : "act_probs"));
            if (mode == "transpose-action")
                graph.Message(1, Node("Transpose", ["action_raw"], "act_probs", IntsAttribute("perm", [1, 0])));
        }
        graph.Message(12, Value("logits", 1, [-1, -1]));
        graph.Message(12, Value("act_probs", 1, [-1, -1]));
        var model = new Proto();
        model.Number(1, 8);
        model.Text(2, "DecisionInference.Tests");
        model.Message(7, graph);
        var opset = new Proto();
        opset.Number(2, 13);
        model.Message(8, opset);
        return model.Bytes();
    }

    private static Proto Value(string name, int type, long[] dimensions)
    {
        var shape = new Proto();
        for (int i = 0; i < dimensions.Length; i++)
        {
            var dimension = new Proto();
            if (dimensions[i] < 0)
                dimension.Text(2, $"{name}_d{i}");
            else dimension.Number(1, dimensions[i]);
            shape.Message(1, dimension);
        }
        var tensor = new Proto();
        tensor.Number(1, type);
        tensor.Message(2, shape);
        var typeProto = new Proto();
        typeProto.Message(1, tensor);
        var value = new Proto();
        value.Text(1, name);
        value.Message(2, typeProto);
        return value;
    }

    private static Proto Node(string op, string[] inputs, string output, params Proto[] attributes)
    {
        var node = new Proto();
        foreach (string input in inputs) node.Text(1, input);
        node.Text(2, output);
        node.Text(4, op);
        foreach (var attribute in attributes) node.Message(5, attribute);
        return node;
    }

    private static Proto IntAttribute(string name, long value)
    {
        var attribute = new Proto();
        attribute.Text(1, name);
        attribute.Number(3, value);
        attribute.Number(20, 2); // AttributeProto.INT
        return attribute;
    }

    private static Proto IntsAttribute(string name, long[] values)
    {
        var attribute = new Proto();
        attribute.Text(1, name);
        foreach (long value in values) attribute.Number(8, value);
        attribute.Number(20, 7); // AttributeProto.INTS
        return attribute;
    }

    private static Proto FloatTensor(string name, long[] dims, float[] values)
    {
        var tensor = new Proto();
        foreach (long dimension in dims) tensor.Number(1, dimension);
        tensor.Number(2, 1); // TensorProto.FLOAT
        tensor.Text(8, name);
        tensor.Data(9, values.SelectMany(BitConverter.GetBytes).ToArray()); // little-endian raw_data
        return tensor;
    }

    private static Proto IntTensor(string name, long[] values)
    {
        var tensor = new Proto();
        tensor.Number(1, values.Length);
        tensor.Number(2, 7); // TensorProto.INT64
        tensor.Text(8, name);
        tensor.Data(9, values.SelectMany(BitConverter.GetBytes).ToArray());
        return tensor;
    }

    private sealed class Proto
    {
        private readonly MemoryStream _stream = new();
        public byte[] Bytes() => _stream.ToArray();
        public void Number(int field, long value) { Varint((ulong)(field << 3)); Varint((ulong)value); }
        public void Text(int field, string value) => Data(field, Encoding.UTF8.GetBytes(value));
        public void Message(int field, Proto value) => Data(field, value.Bytes());
        public void Data(int field, byte[] value)
        {
            Varint((ulong)((field << 3) | 2));
            Varint((ulong)value.Length);
            _stream.Write(value);
        }
        private void Varint(ulong value)
        {
            while (value >= 128) { _stream.WriteByte((byte)((value & 127) | 128)); value >>= 7; }
            _stream.WriteByte((byte)value);
        }
    }
}
