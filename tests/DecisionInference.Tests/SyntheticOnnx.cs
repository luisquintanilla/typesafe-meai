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

    internal static byte[] CreateExternalData(bool julia)
    {
        var tensor = new Proto();
        tensor.Number(1, 1);
        tensor.Number(1, 2);
        tensor.Number(2, 1);
        tensor.Text(8, "external_logits");
        var location = new Proto();
        location.Text(1, "location");
        location.Text(2, "weights.bin");
        tensor.Message(13, location);
        tensor.Number(14, 1);
        var graph = new Proto();
        graph.Text(2, "external-data-loading-fixture");
        foreach (var input in Inputs()) graph.Message(11, Value(input.Name, input.ElementType, input.Shape));
        graph.Message(5, tensor);
        graph.Message(1, Node("Identity", ["external_logits"], "logits"));
        graph.Message(12, Value("logits", 1, [1, 2]));
        if (!julia)
        {
            graph.Message(5, FloatTensor("actions", [1, 2], [.5f, .5f]));
            graph.Message(1, Node("Identity", ["actions"], "act_probs"));
            graph.Message(12, Value("act_probs", 1, [1, 2]));
        }
        var model = new Proto();
        model.Number(1, 8);
        model.Message(7, graph);
        var opset = new Proto();
        opset.Number(2, 13);
        model.Message(8, opset);
        return model.Bytes();
    }

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

    // Julia-only logits fixture. The existing two-output Create modes are deliberately unchanged.
    internal static byte[] CreateJulia(Input[]? inputs = null, string outputName = "logits",
        int outputType = 1, long[]? outputShape = null, float[]? values = null, bool transpose = false, bool probeInputs = false)
    {
        var graph = new Proto();
        graph.Text(2, "synthetic-julia-fixture");
        foreach (var input in inputs ?? Inputs())
            graph.Message(11, Value(input.Name, input.ElementType, input.Shape));
        string source = "marker_pos";
        if (probeInputs)
        {
            // Exact integral signature of all five input buffers; no reductions with order-dependent rounding.
            foreach (string name in new[] { "input_ids", "attention_mask", "marker_mask", "qtype" })
                graph.Message(1, Node("Cast", [name], name + "_float", IntAttribute("to", 1)));
            graph.Message(1, Node("GatherElements", ["input_ids_float", "marker_pos"], "ids_at_markers", IntAttribute("axis", 1)));
            graph.Message(1, Node("GatherElements", ["attention_mask_float", "marker_pos"], "attention_at_markers", IntAttribute("axis", 1)));
            graph.Message(5, FloatTensor("ten", [], [10]));
            graph.Message(5, FloatTensor("hundred", [], [100]));
            graph.Message(5, FloatTensor("thousand", [], [1000]));
            graph.Message(5, IntTensor("unsqueeze_axis", [1]));
            graph.Message(1, Node("Mul", ["ids_at_markers", "ten"], "weighted_ids"));
            graph.Message(1, Node("Mul", ["marker_mask_float", "hundred"], "weighted_mask"));
            graph.Message(1, Node("Mul", ["qtype_float", "thousand"], "weighted_type"));
            graph.Message(1, Node("Unsqueeze", ["weighted_type", "unsqueeze_axis"], "type_column"));
            graph.Message(1, Node("Add", ["weighted_ids", "attention_at_markers"], "sum_one"));
            graph.Message(1, Node("Add", ["sum_one", "weighted_mask"], "sum_two"));
            graph.Message(1, Node("Add", ["sum_two", "type_column"], "input_signature"));
            source = "input_signature";
        }
        if (values is not null)
        {
            source = "constant_logits";
            graph.Message(5, FloatTensor(source, outputShape ?? [1, 2], values));
        }
        graph.Message(1, Node("Cast", [source], transpose ? "untransposed" : outputName, IntAttribute("to", outputType)));
        if (transpose)
            graph.Message(1, Node("Transpose", ["untransposed"], outputName, IntsAttribute("perm", [1, 0])));
        graph.Message(12, Value(outputName, outputType, outputShape ?? [-1, -1]));
        var model = new Proto();
        model.Number(1, 8);
        model.Text(2, "DecisionInference.Tests.Julia");
        model.Message(7, graph);
        var opset = new Proto();
        opset.Number(2, 13);
        model.Message(8, opset);
        return model.Bytes();
    }

    internal static byte[] CreateLayaOutputs(float[] logits, float[] actions, int rows, int width,
        int actionWidth = 2, bool extraOutput = false)
    {
        var graph = new Proto();
        graph.Text(2, "laya-output-policy-fixture");
        foreach (var input in Inputs()) graph.Message(11, Value(input.Name, input.ElementType, input.Shape));
        graph.Message(5, FloatTensor("logits_value", [rows, width], logits));
        graph.Message(5, FloatTensor("action_value", [rows, actionWidth], actions));
        graph.Message(1, Node("Identity", ["logits_value"], "logits"));
        graph.Message(1, Node("Identity", ["action_value"], "act_probs"));
        graph.Message(12, Value("logits", 1, [rows, width]));
        graph.Message(12, Value("act_probs", 1, [rows, actionWidth]));
        if (extraOutput)
        {
            graph.Message(1, Node("Identity", ["logits_value"], "unused_output"));
            graph.Message(12, Value("unused_output", 1, [rows, width]));
        }
        var model = new Proto();
        model.Number(1, 8);
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
