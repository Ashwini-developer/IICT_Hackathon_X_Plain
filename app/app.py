#!/usr/bin/env python3
import os, sys, subprocess, json, streamlit as st, pandas as pd
from pathlib import Path

# --- Fix: Add ../python to sys.path ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, "python")
DUMPS = os.path.join(ROOT, "dumps")
MODEL_DIR = os.path.join(ROOT, "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "mobilenetv2.onnx")

if PY not in sys.path:
    sys.path.insert(0, PY)

os.makedirs(DUMPS, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Detect TVM availability
try:
    import tvm  # noqa: F401
    TVM_AVAILABLE = True
except Exception:
    TVM_AVAILABLE = False

st.set_page_config(layout="wide", page_title="X-Plain CPU Edition")
st.title("X-Plain — Explainable Compiler (CPU Edition)")

col1, col2 = st.columns([1, 1])

# LEFT: export/upload
with col1:
    st.header("Model Export / Upload")
    if st.button("Export Default Model (MobileNetV2)"):
        proc = subprocess.run([sys.executable, os.path.join(PY, "export_model.py")],
                              capture_output=True, text=True)
        st.text(proc.stdout)
        if proc.stderr:
            st.text(proc.stderr)

    uploaded = st.file_uploader("Upload your ONNX model", type=["onnx"])
    model_path = DEFAULT_MODEL
    uploaded_flag = False
    if uploaded:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "uploaded.onnx")
        with open(model_path, "wb") as f:
            f.write(uploaded.getbuffer())
        uploaded_flag = True
        st.success(f"Uploaded model saved to {model_path}")

    # show MLIR if present (torch-mlir optional)
    for fname in ["torch.mlir", "linalg.mlir"]:
        p = os.path.join(DUMPS, fname)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
            with st.expander(fname):
                st.code(txt[:3000] + ("\n...truncated" if len(txt) > 3000 else ""), language="mlir")

# RIGHT: benchmarking and quantization options
with col2:
    st.header("Benchmarking & Precision")
    precision_mode = st.radio("Precision mode", ["FP32 only", "INT8 only", "FP32 vs INT8 (side-by-side)"], index=2)
    st.markdown("Backends shown depend on whether TVM is installed on this machine.")

    run_button = st.button("Run Comparison on Selected Model")
    quantize_script = os.path.join(PY, "quantize_model.py")

    if run_button:
        # choose model paths
        fp32_path = model_path
        int8_path = os.path.join(MODEL_DIR, "mobilenetv2_int8.onnx") if not uploaded_flag else os.path.join(MODEL_DIR, "uploaded_int8.onnx")

        # if INT8 needed, create quantized model
        if precision_mode in ["INT8 only", "FP32 vs INT8 (side-by-side)"]:
            if not os.path.exists(int8_path):
                try:
                    proc_q = subprocess.run([sys.executable, quantize_script, fp32_path, int8_path], capture_output=True, text=True)
                    st.text(proc_q.stdout)
                    if proc_q.stderr:
                        st.text(proc_q.stderr)
                except Exception as e:
                    st.error(f"Quantization failed: {e}")
                    int8_path = None

        # Determine which args to pass to run_comparison
        run_comp = os.path.join(PY, "run_comparison.py")
        args = [sys.executable, run_comp]
        if precision_mode == "FP32 only":
            args.append(fp32_path)
        elif precision_mode == "INT8 only":
            args.append(fp32_path); args.append(int8_path)
        else:  # both
            args.append(fp32_path); args.append(int8_path)

        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            st.error("Benchmarking failed. See output:")
            st.text(proc.stdout)
            st.text(proc.stderr)
        else:
            try:
                res = json.loads(proc.stdout)
            except Exception:
                st.text(proc.stdout)
                res = {}
            if res:
                st.subheader("Benchmark Results (JSON)")
                st.json(res)
                df = pd.DataFrame(res).T
                if "latency_ms" in df:
                    st.subheader("Latency (ms)")
                    st.bar_chart(df["latency_ms"])
                if "throughput" in df:
                    st.subheader("Throughput (fps)")
                    st.bar_chart(df["throughput"])

                # Compiler insights
                from compiler_insights import explain_results
                st.subheader("Compiler Insights")
                for insight in explain_results(res):
                    st.markdown(f"- {insight}")

# Model Graph + Relay IR + Diff + Pass Timeline
st.markdown("---")
st.header("Model Graph & Compiler IR")

from graph_visualizer import dump_graph_json, simulate_pass_fusion_graph
from dump_relay import dump_relay_ir, dump_relay_timeline
from ir_diff import diff_ir

import pandas as pd

tab1, tab2, tab3 = st.tabs(["ONNX Graph", "Fusion Graph", "Relay IR"])

with tab1:
    st.subheader("ONNX Graph (Full Model)")
    graph_data = dump_graph_json(model_path)

    # Show summary counts
    st.write("### Operator Counts (Before Optimization)")
    st.json(graph_data["counts"])
    st.session_state["counts_before"] = graph_data["counts"]

    # Optional big graph (expandable)
    with st.expander("Show Full ONNX Graph (Optional)"):
        nodes = json.dumps([{"data": n} for n in graph_data["nodes"]])
        edges = json.dumps([{"data": e} for e in graph_data["edges"]])
        graph_data = dump_graph_json(model_path)
        if not graph_data["nodes"] or not graph_data["edges"]:
            st.error("ONNX graph is empty — check if ONNX model was exported correctly.")
        else:
            # Consolidated counts
            counts_before = {}
            for n in graph_data["nodes"]:
                lbl = n["label"]
                counts_before[lbl] = counts_before.get(lbl, 0) + 1

            st.write("### Operator Counts (Before Optimization)")
            st.json(counts_before)

            # Save counts for comparison in Fusion tab
            st.session_state["counts_before"] = counts_before

            # Cytoscape graph
            nodes = json.dumps([{"data": n} for n in graph_data["nodes"]])
            edges = json.dumps([{"data": e} for e in graph_data["edges"]])
            html = f"""
            <html><head>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
            </head>
            <body><div id="cy" style="width:100%; height:800px;"></div>
            <script>
            var cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: {{
                    nodes: {nodes},
                    edges: {edges}
                }},
                style: [
                    {{ selector: 'node', style: {{ 'content': 'data(label)', 'background-color': 'data(color)', 'color':'#fff','text-valign':'center','text-halign':'center','font-size':10 }} }},
                    {{ selector: 'edge', style: {{ 'width': 2, 'line-color': '#ccc','target-arrow-shape': 'triangle' }} }}
                ],
                layout: {{ name: 'cose', padding: 10 }}
            }});
            </script></body></html>
            """
            st.components.v1.html(html, height=800, scrolling=True)
  

with tab2:
    st.subheader("Simulated Pass Fusion Graph")
    graph_data = simulate_pass_fusion_graph(model_path)

    # Show summary counts
    st.write("### Operator Counts (After Fusion)")
    st.json(graph_data["counts"])

    if "counts_before" in st.session_state:
        before = st.session_state["counts_before"]
        after = graph_data["counts"]

        conv_before = before.get("Conv", 0) + before.get("MatMul/Gemm", 0)
        fused_after = after.get("FusedOp", 0)

        st.success(f"Before: {conv_before} Conv/MatMul ops → After: {fused_after} FusedOps")

        if conv_before > 0:
            reduction = (1 - fused_after / conv_before) * 100
            st.info(f"That’s a {reduction:.1f}% reduction in heavy ops.")

        # Bar chart comparison
        import pandas as pd
        df = pd.DataFrame({"Before": before, "After": after}).fillna(0).astype(int)
        st.bar_chart(df)

    # Optional big graph
    with st.expander("Show Full Fusion Graph (Optional)"):
        nodes = json.dumps([{"data": n} for n in graph_data["nodes"]])
        edges = json.dumps([{"data": e} for e in graph_data["edges"]])
        graph_data = simulate_pass_fusion_graph(model_path)
        if not graph_data["nodes"] or not graph_data["edges"]:
            st.error("Fusion graph is empty — check simulation output.")
        else:
            # Consolidated counts
            counts_after = {}
            fused_ops = 0
            for n in graph_data["nodes"]:
                lbl = n["label"]
                counts_after[lbl] = counts_after.get(lbl, 0) + 1
                if lbl == "FusedOp":
                    fused_ops += 1

            st.write("### Operator Counts (After Fusion)")
            st.json(counts_after)
            st.info(f"Optimization fused Conv/MatMul nodes into {fused_ops} FusedOp nodes.")

            # Side-by-side bar chart
            if "counts_before" in st.session_state:
                before = st.session_state["counts_before"]
                df = pd.DataFrame({
                    "Before Fusion": before,
                    "After Fusion": counts_after
                }).fillna(0).astype(int)
                st.write("### Operator Count Comparison")
                st.bar_chart(df)

            # Cytoscape graph
            nodes = json.dumps([{"data": n} for n in graph_data["nodes"]])
            edges = json.dumps([{"data": e} for e in graph_data["edges"]])
            html = f"""
            <html><head>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
            </head>
            <body><div id="cy2" style="width:100%; height:800px;"></div>
            <script>
            var cy = cytoscape({{
                container: document.getElementById('cy2'),
                elements: {{
                    nodes: {nodes},
                    edges: {edges}
                }},
                style: [
                    {{ selector: 'node', style: {{ 'content': 'data(label)', 'background-color': 'data(color)', 'color':'#fff','text-valign':'center','text-halign':'center','font-size':10 }} }},
                    {{ selector: 'edge', style: {{ 'width': 2, 'line-color': '#ccc','target-arrow-shape': 'triangle' }} }}
                ],
                layout: {{ name: 'cose', padding: 10 }}
            }});
            </script></body></html>
            """
            st.components.v1.html(html, height=800, scrolling=True)
    


with tab3:
    st.subheader("TVM Relay IR (Before/After)")
    if st.button("Dump Relay IR (raw & optimized)"):
        paths = dump_relay_timeline(model_path)
        st.success("Relay IR dumps created.")
        for p in paths:
            if os.path.exists(p):
                st.markdown(f"**{os.path.basename(p)}**")
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                st.code(content[:2000] + ("\n...truncated" if len(content) > 2000 else ""), language="text")

    if st.button("Compare Relay IR (diff)"):
        raw = dump_relay_ir(model_path, optimized=False, opt_level=0)
        opt = dump_relay_ir(model_path, optimized=True, opt_level=3)
        if raw and opt and os.path.exists(raw) and os.path.exists(opt):
            with open(raw, "r", encoding="utf-8") as f:
                raw_txt = f.read()
            with open(opt, "r", encoding="utf-8") as f:
                opt_txt = f.read()
            diff_txt = diff_ir(raw_txt, opt_txt, context=3)
            if diff_txt.strip():
                st.subheader("Relay IR Diff")
                st.code(diff_txt, language="diff")
            else:
                st.info("No differences detected.")
        else:
            st.warning("Relay dumps not available (TVM missing?)")
