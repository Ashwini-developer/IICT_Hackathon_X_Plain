#!/usr/bin/env python3
import os, time, numpy as np, psutil
try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime is required. Install with `pip install onnxruntime`.") from e

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DEFAULT = os.path.join(ROOT, "models", "mobilenetv2.onnx")

def benchmark_onnx(model_path=None, input_shape=(1,3,224,224), iters=30, warmup=5):
    if model_path is None:
        model_path = MODEL_DEFAULT
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}. Run export_model.py first.")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(*input_shape).astype(np.float32)
    feeds = {sess.get_inputs()[0].name: x}

    for _ in range(warmup):
        sess.run(None, feeds)

    times=[]
    proc=psutil.Process(os.getpid())
    start_mem=proc.memory_info().rss
    for _ in range(iters):
        t0=time.perf_counter()
        sess.run(None, feeds)
        times.append(time.perf_counter()-t0)
    peak_mem=proc.memory_info().rss

    import numpy as _np
    latency_ms = float(_np.median(times)*1000.0)
    throughput = float(1.0 / _np.mean(times))
    memory_mb = float((peak_mem-start_mem)/(1024.0*1024.0))
    energy_est = float(_np.median(times)*10.0)
    return {"latency_ms": latency_ms, "throughput": throughput, "memory_mb": memory_mb, "energy_est": energy_est}

if __name__ == "__main__":
    print(benchmark_onnx())
