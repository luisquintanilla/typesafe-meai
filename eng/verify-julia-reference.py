"""Independent pinned Python sequence/Collator/typed API + Python ORT CPU verification.

No model downloads. The six recorded PyTorch comparisons are upstream evidence,
not a local PyTorch reproduction or quality benchmark.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import onnxruntime as ort
import tokenizers
import torch
import transformers
from transformers import AutoTokenizer

directory, receipt_path = map(Path, sys.argv[1:3])
expected_hashes = {
    "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    "tokenizer_config.json": "6b069e57db0ce0794c22547725275f22618809c4ef4249307337e651a0dfef8c",
    "model.onnx": "97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9",
    "model.onnx.data": "fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1",
    "julia/data.py": "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e",
    "julia/typed.py": "ed89e66a70fcedac1339347bd8fcca69fd2cd1a31535ad0148dae42c610b544d",
    "parity-cases.json": "534670f25823f3f9abf9e812500918fabc606682dfd4e2456d827889d9bdca20",
}
for name, expected in expected_hashes.items():
    with (directory / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
package = types.ModuleType("julia")
package.__path__ = [str(directory / "julia")]
sys.modules["julia"] = package
for name in ("data", "typed"):
    spec = importlib.util.spec_from_file_location("julia." + name, directory / "julia" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from julia.data import sequence, Collator
from julia.typed import predict_typed

tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
managed = json.loads(receipt_path.read_text(encoding="utf-8"))
assert managed["revision"] == "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8"
assert len(managed["requests"]) == 4, "Expected three all-kind requests and one six-row upstream subset"
assert managed["concurrentCalls"] == 2 and managed["preCancelled"] is True
assert managed["rejectionRecovery"] is True and managed["ownedSnapshots"] is True
assert isinstance(managed["inFlightOnnxCancellation"], str) and managed["inFlightOnnxCancellation"]
max_logit_error = max_probability_error = max_decoder_error = max_score_error = 0.0
rows_compared = tensors_compared = 0


class Engine:
    def __init__(self, evidence):
        self.evidence = evidence

    def logits(self, rows):
        global max_logit_error, tensors_compared
        for row in rows:
            row["_encoded"] = sequence(tokenizer, row, 1024, 256, strict=True)
        batch = Collator(tokenizer, 1024, 256)(rows, include_targets=False)
        feed = {name: tensor.numpy() for name, tensor in batch.items()}
        b, length, width = self.evidence["batch"], self.evidence["length"], self.evidence["width"]
        expected_shapes = {"input_ids": (b, length), "attention_mask": (b, length),
                           "marker_pos": (b, width), "marker_mask": (b, width), "qtype": (b,)}
        for name, value in feed.items():
            assert value.shape == expected_shapes[name], (name, value.shape, expected_shapes[name])
            assert len(self.evidence[name]) == value.size, (name, "element count")
            np.testing.assert_array_equal(value.flatten(), self.evidence[name], err_msg=name)
            tensors_compared += 1
        actual = session.run(["logits"], feed)[0]
        managed_logits = np.array(self.evidence["logits"], dtype=np.float32).reshape(actual.shape)
        result = []
        for i, row in enumerate(rows):
            count = len(row["options"])
            error = float(np.max(np.abs(actual[i, :count] - managed_logits[i, :count])))
            max_logit_error = max(max_logit_error, error)
            assert error <= 1e-4, ("logits", error)
            result.append(actual[i, :count].astype(float).tolist())
        return result


for index, evidence in enumerate(managed["requests"]):
    count = 3 if index < 3 else 6
    assert evidence["batch"] == len(evidence["questions"]) == len(evidence["answers"]) == count
    assert [a["id"] for a in evidence["answers"]] == list(evidence["questions"]), "Answer order/correlation"
    kinds = [q["type"] for q in evidence["questions"].values()]
    assert kinds == (["choice", "noul", "score"] if index < 3 else ["choice"] * 6), kinds
    assert 0 < evidence["length"] <= 1024 and evidence["length"] % 8 == 0
    assert 2 <= evidence["width"] <= 20
    assert len(evidence["logits"]) == evidence["batch"] * evidence["width"]
    expected = predict_typed(Engine(evidence), evidence["state"], evidence["questions"])["answers"]
    assert list(expected) == list(evidence["questions"])
    z = np.array(evidence["logits"], dtype=np.float32).reshape(evidence["batch"], evidence["width"])
    for row, actual in enumerate(evidence["answers"]):
        answer = expected[actual["id"]]
        probabilities = list(answer["probabilities"].values())
        assert len(actual["probabilities"]) == len(probabilities), (actual["id"], "distribution cardinality")
        maximum = max(map(float, z[row, :len(probabilities)]))
        exp = [math.exp(float(v) - maximum) for v in z[row, :len(probabilities)]]
        total = sum(exp)
        decoded = [v / total for v in exp]
        max_decoder_error = max(max_decoder_error, max(abs(a - b) for a, b in zip(actual["probabilities"], decoded)))
        error = max(abs(a - b) for a, b in zip(actual["probabilities"], probabilities))
        max_probability_error = max(max_probability_error, error)
        assert error <= 5e-5, ("independent probabilities", error)
        assert max_decoder_error <= 1e-12, ("full softmax from identical logits", max_decoder_error)
        assert abs(sum(actual["probabilities"]) - 1) <= 1e-12
        if answer["type"] == "choice":
            assert actual["selected"] == answer["choice"]
        if answer["type"] == "score":
            error = abs(actual["expectedIndex"] - answer["score"])
            max_score_error = max(max_score_error, error)
            assert error <= 1e-4 * (len(probabilities) - 1)
        rows_compared += 1

assert rows_compared == 15 and tensors_compared == 20, "Incomplete proof"
summary = {
    "rows": rows_compared, "tensorsExactlyEqual": tensors_compared,
    "maxAbsLogitError": max_logit_error, "logitTolerance": 1e-4,
    "maxAbsProbabilityError": max_probability_error, "probabilityTolerance": 5e-5,
    "maxAbsDecoderError": max_decoder_error, "sameLogitsSoftmaxTolerance": 1e-12,
    "maxAbsScoreError": max_score_error, "scoreTolerance": "1e-4 * (K-1)",
    "recipe": "unmodified pinned Python sequence(strict=True,1024,256), Collator and predict_typed",
    "independentExecution": "Python ORT CPU on independently encoded inputs; same pinned ONNX weights",
    "pytorch": "6 upstream-recorded cases checked by .NET host; no local PyTorch weight execution",
    "versions": {"ort": ort.__version__, "tokenizers": tokenizers.__version__,
                 "transformers": transformers.__version__, "torch": torch.__version__, "numpy": np.__version__},
    "managedReceiptSha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "verifiedAssets": expected_hashes,
}
output = receipt_path.with_name("julia-reference.json")
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
