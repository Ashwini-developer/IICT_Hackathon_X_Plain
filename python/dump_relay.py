#!/usr/bin/env python3
import os, onnx

try:
    import tvm
    from tvm import relay
except Exception:
    tvm = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DEFAULT = os.path.join(ROOT, "models", "mobilenetv2.onnx")
DUMPS = os.path.join(ROOT, "dumps")
os.makedirs(DUMPS, exist_ok=True)

def dump_relay_ir(model_path=MODEL_DEFAULT, optimized=False, opt_level=3, output_path=None):
    """
    Dumps Relay IR. If optimized True, uses PassContext(opt_level=opt_level).
    Also supports generating a pass timeline: opt_level 0 (raw), 1, 2, 3.
    """
    if tvm is None:
        # simulate dump so UI can proceed (makes a simple placeholder)
        txt = f"# TVM not installed. Simulated Relay IR ({'optimized' if optimized else 'raw'}, opt_level={opt_level})\n"
        txt += "def func():\n    # simulated ops\n    conv = nn.conv2d(...)\n    bn = nn.batch_norm(conv)\n    relu = nn.relu(bn)\n"
        out = output_path if output_path else os.path.join(DUMPS, f"relay_ir_sim_opt{opt_level}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("[dump_relay] TVM not installed — wrote simulated relay IR to", out)
        return out

    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    shape_dict = {input_name: (1, 3, 224, 224)}
    mod, params = relay.frontend.from_onnx(model, shape_dict)

    if optimized:
        with tvm.transform.PassContext(opt_level=opt_level):
            # Use relay.build to apply passes (but we do not need lib)
            optimized_mod = relay.optimize(mod, target="llvm", params=params)[0]
            ir_txt = str(optimized_mod)
    else:
        ir_txt = str(mod)

    if output_path is None:
        fname = f"relay_ir_opt{opt_level}.txt" if optimized else "relay_ir_raw.txt"
        output_path = os.path.join(DUMPS, fname)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ir_txt)
    print(f"[dump_relay] Wrote Relay IR to {output_path} (optimized={optimized}, opt_level={opt_level})")
    return output_path

def dump_relay_timeline(model_path=MODEL_DEFAULT):
    """
    Dumps a sequence of Relay IR snapshots for a pass timeline: raw, opt1, opt2, opt3
    Returns list of paths [raw, opt1, opt2, opt3]
    """
    paths = []
    paths.append(dump_relay_ir(model_path, optimized=False, opt_level=0, output_path=os.path.join(DUMPS, "relay_raw.txt")))
    for lvl in [1,2,3]:
        paths.append(dump_relay_ir(model_path, optimized=True, opt_level=lvl, output_path=os.path.join(DUMPS, f"relay_opt{lvl}.txt")))
    return paths

if __name__ == "__main__":
    dump_relay_timeline()
