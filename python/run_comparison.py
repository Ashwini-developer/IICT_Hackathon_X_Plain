#!/usr/bin/env python3
import os, json, sys
from benchmark_onnx import benchmark_onnx

# Try TVM imports
try:
    from benchmark_tvm import benchmark_tvm
    from benchmark_tvm_ryzen import benchmark_tvm_ryzen
    TVM_AVAILABLE = True
except Exception:
    TVM_AVAILABLE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DEFAULT = os.path.join(ROOT, "models", "mobilenetv2.onnx")
MODEL_INT8_DEFAULT = os.path.join(ROOT, "models", "mobilenetv2_int8.onnx")

def run_all(fp32_model=MODEL_DEFAULT, int8_model=MODEL_INT8_DEFAULT):
    results = {}
    # FP32
    try:
        results["FP32-ONNXRuntime"] = benchmark_onnx(fp32_model)
    except Exception as e:
        results["FP32-ONNXRuntime"] = {"error": str(e)}
    if TVM_AVAILABLE:
        try:
            results["FP32-TVM-CortexA75"] = benchmark_tvm(fp32_model)
        except Exception as e:
            results["FP32-TVM-CortexA75"] = {"error": str(e)}
        try:
            results["FP32-TVM-Ryzen"] = benchmark_tvm_ryzen(fp32_model)
        except Exception as e:
            results["FP32-TVM-Ryzen"] = {"error": str(e)}
    # INT8 (only if file exists)
    if os.path.exists(int8_model):
        try:
            results["INT8-ONNXRuntime"] = benchmark_onnx(int8_model)
        except Exception as e:
            results["INT8-ONNXRuntime"] = {"error": str(e)}
        if TVM_AVAILABLE:
            try:
                results["INT8-TVM-CortexA75"] = benchmark_tvm(int8_model)
            except Exception as e:
                results["INT8-TVM-CortexA75"] = {"error": str(e)}
            try:
                results["INT8-TVM-Ryzen"] = benchmark_tvm_ryzen(int8_model)
            except Exception as e:
                results["INT8-TVM-Ryzen"] = {"error": str(e)}
    return results

if __name__ == "__main__":
    fp32 = sys.argv[1] if len(sys.argv) > 1 else MODEL_DEFAULT
    int8 = sys.argv[2] if len(sys.argv) > 2 else MODEL_INT8_DEFAULT
    res = run_all(fp32, int8)
    print(json.dumps(res, indent=2))
