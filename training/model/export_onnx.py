"""
export_onnx.py

Usage:
    python export_onnx.py
"""

import os
import sys
import torch
import torch.nn as nn
import timm
import onnx
import onnxruntime as ort

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def build_model() -> nn.Module:
    model = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=0)
    in_features = model.num_features
    model.classifier = nn.Sequential(nn.Dropout(p=config.DROPOUT), nn.Linear(in_features, 2))
    return model


def export():
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Load model
    model = build_model()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # Dummy input
    dummy_input = torch.randn(1, 3, config.INPUT_SIZE, config.INPUT_SIZE)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        config.ONNX_MODEL_PATH,
        opset_version=config.ONNX_OPSET_VERSION,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[INFO] ONNX model exported to: {config.ONNX_MODEL_PATH}")

    # Validate ONNX model structure
    onnx_model = onnx.load(config.ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model structure check passed.")

    # Validate inference with ONNX Runtime
    session = ort.InferenceSession(
        config.ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    dummy_np = dummy_input.numpy()
    outputs = session.run(None, {input_name: dummy_np})
    print(f"[INFO] ONNX Runtime inference test passed. Output shape: {outputs[0].shape}")
    print(f"[DONE] Export complete: {config.ONNX_MODEL_PATH}")


if __name__ == "__main__":
    export()
