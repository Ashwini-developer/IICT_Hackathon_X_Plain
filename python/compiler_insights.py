#!/usr/bin/env python3
def explain_results(results):
    insights = []
    # ONNX FP32 vs INT8
    if "FP32-ONNXRuntime" in results and "INT8-ONNXRuntime" in results:
        a = results["FP32-ONNXRuntime"].get("latency_ms", None)
        b = results["INT8-ONNXRuntime"].get("latency_ms", None)
        if a and b:
            gain = (a - b) / a * 100.0
            insights.append(f"ONNXRuntime: INT8 reduced latency by ~{gain:.1f}% (FP32 {a:.1f}ms → INT8 {b:.1f}ms).")
    # TVM examples
    if "FP32-TVM-Ryzen" in results and "INT8-TVM-Ryzen" in results:
        a = results["FP32-TVM-Ryzen"].get("throughput", None)
        b = results["INT8-TVM-Ryzen"].get("throughput", None)
        if a and b:
            boost = (b - a) / a * 100.0
            insights.append(f"TVM Ryzen: INT8 increased throughput by ~{boost:.1f}% (vectorized kernels / fuse).")
    if not insights:
        insights.append("No detailed insights available. Try running both FP32 and INT8 or enable TVM.")
    return insights
