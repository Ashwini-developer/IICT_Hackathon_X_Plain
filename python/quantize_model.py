#!/usr/bin/env python3
import sys, os
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

class DummyDataReader(CalibrationDataReader):
    def __init__(self, model_path):
        # Minimal dummy input reader for calibration
        self.model = onnx.load(model_path)
        self.enum_data_dicts = None

    def get_next(self):
        return None

def quantize_model(fp32_model_path, int8_model_path):
    if not os.path.exists(fp32_model_path):
        print(f"[quantize_model] ERROR: {fp32_model_path} not found")
        return

    print(f"[quantize_model] Quantizing {fp32_model_path} -> {int8_model_path}")

    # Use static quantization with QLinear ops
    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=DummyDataReader(fp32_model_path),
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul"],  # QLinearConv + QLinearMatMul
        per_channel=True,
        reduce_range=False
    )

    print(f"[quantize_model] Quantized model written to {int8_model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python quantize_model.py <fp32_model.onnx> <int8_model.onnx>")
        sys.exit(1)

    fp32_model = sys.argv[1]
    int8_model = sys.argv[2]
    quantize_model(fp32_model, int8_model)
