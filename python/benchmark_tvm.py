#!/usr/bin/env python3
import os, time, numpy as np, psutil, onnx
try:
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
except Exception:
    tvm=None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DEFAULT = os.path.join(ROOT, "models", "mobilenetv2.onnx")

def benchmark_tvm(model_path=None, input_shape=(1,3,224,224), iters=30, warmup=5):
    if tvm is None:
        return {"error":"TVM not installed"}
    if model_path is None:
        model_path = MODEL_DEFAULT
    if not os.path.exists(model_path):
        return {"error":f"ONNX model not found at {model_path}"}

    onnx_model = onnx.load(model_path)
    input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    target = "llvm -mcpu=cortex-a75"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    dev = tvm.cpu()
    m = graph_executor.GraphModule(lib["default"](dev))
    x = np.random.randn(*input_shape).astype("float32")
    m.set_input(input_name, tvm.nd.array(x))
    for _ in range(warmup): m.run()
    times=[]; proc=psutil.Process(os.getpid()); start_mem=proc.memory_info().rss
    for _ in range(iters):
        t0=time.perf_counter(); m.run(); times.append(time.perf_counter()-t0)
    peak_mem=proc.memory_info().rss
    latency_ms=float(np.median(times)*1000); throughput=float(1.0/np.mean(times))
    memory_mb=float((peak_mem-start_mem)/(1024*1024)); energy_est=float(np.median(times)*8)
    return {"latency_ms":latency_ms,"throughput":throughput,"memory_mb":memory_mb,"energy_est":energy_est}

if __name__ == "__main__":
    print(benchmark_tvm())
