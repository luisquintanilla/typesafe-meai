"""Optional independent local reference: HF tokenizers + NumPy + ONNX Runtime.

No downloads, transformers, remote code, or production dependency. Supply the
hash-verified local Laya directory and the `stages` command's JSON receipt.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def main():
    root, receipt_path = map(Path, sys.argv[1:3])
    managed = json.loads(receipt_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(root / "tokenizer" / "tokenizer.json"))
    config = json.loads((root / "laya_config.json").read_text())
    state = "Email: Payroll asks for your password on a non-company sign-in page."
    specs = [
        ("choice", "Choose one option.", ["0: Legitimate", "1: Spam", "2: Phishing"]),
        ("noul", "Does the email request a password?",
         ["false: no, the statement does not hold", "true: yes, the statement holds"]),
        ("score", "Assess the risk in the email.",
         ["level 0: low", "level 1: medium", "level 2: high"]),
    ]
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    cls, sep, mask, pad = [tokenizer.token_to_id(s) for s in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]]
    rows, positions = [], []
    for kind, instruction, options in specs:
        head = encode(f"{kind} question: {instruction}")
        options = [encode(" " + value) for value in options]
        assert all(len(value) <= 49 for value in options)
        remaining = config["head_max_len"] - sum(len(value) + 1 for value in options)
        assert remaining >= 16 and len(head) <= remaining
        row, markers = [cls] + head + [sep], []
        for value in options:
            markers.append(len(row))
            row += [mask] + value
        row += [sep] + encode(state) + [sep]
        assert len(row) <= config["max_len"]
        rows.append(row)
        positions.append(markers)
    batch, length, width = len(rows), max(map(len, rows)), max(map(len, positions))
    inputs = {
        "input_ids": np.full((batch, length), pad, dtype=np.int64),
        "attention_mask": np.zeros((batch, length), dtype=np.int64),
        "marker_pos": np.zeros((batch, width), dtype=np.int64),
        "marker_mask": np.zeros((batch, width), dtype=bool),
        "qtype": np.array([0, 2, 1], dtype=np.int64),
    }
    for i, (row, markers) in enumerate(zip(rows, positions)):
        inputs["input_ids"][i, :len(row)] = row
        inputs["attention_mask"][i, :len(row)] = 1
        inputs["marker_pos"][i, :len(markers)] = markers
        inputs["marker_mask"][i, :len(markers)] = True
    for key, value in inputs.items():
        np.testing.assert_array_equal(value.reshape(-1), managed[key], err_msg=key)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(root / "laya.onnx"), options, providers=["CPUExecutionProvider"])
    logits, action = session.run(["logits", "act_probs"], inputs)
    np.testing.assert_allclose(logits.reshape(-1), managed["logits"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(action.reshape(-1), managed["act_probs"], rtol=0, atol=1e-5)
    decoded = []
    for row, (kind, _, choices) in enumerate(specs):
        count = len(choices)
        raw_temperature = config["temperature_by_options"].get(
            f"{kind}:{'2' if count == 2 else '3-5'}",
            config["temperature"][{"choice": 0, "score": 1, "noul": 2}[kind]])
        temperature = min(5.0, max(0.5, raw_temperature))
        values = logits[row, :count].astype(np.float64)
        weights = np.exp((values - np.max(values)) / temperature)
        probabilities = weights / weights.sum()
        np.testing.assert_allclose(probabilities, managed["probabilities"][row], rtol=0, atol=1e-5)
        np.testing.assert_allclose(action[row], managed["evidence"][row]["actionProbabilities"], rtol=0, atol=1e-5)
        if kind == "score":
            np.testing.assert_allclose(np.dot(np.arange(count), probabilities), managed["expectedIndex"], rtol=0, atol=1e-5)
        decoded.append({"kind": kind, "probabilities": probabilities.tolist(),
                        "expectedIndex": float(np.dot(np.arange(count), probabilities)) if kind == "score" else None})
    print(json.dumps({"reference": "independent HF tokenizers/NumPy/ORT CPU",
                      "ortVersion": ort.__version__, "inputShapes": [batch, length, width],
                      "preparation": "exact five-tensor equality", "logitsTolerance": 1e-4,
                      "probabilityTolerance": 1e-5, "results": decoded}, indent=2))


if __name__ == "__main__":
    main()
