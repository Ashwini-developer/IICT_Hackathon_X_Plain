#!/usr/bin/env python3
import onnx, networkx as nx

# Ops to hide (they clutter the graph but don’t add much meaning)
TRIVIAL_OPS = {"Identity", "Dropout", "Constant", "Reshape", "Flatten"}

# Colors for semantic node types
COLOR_MAP = {
    "Conv": "#4B9CD3",       # Blue
    "MatMul": "#E74C3C",     # Red
    "Gemm": "#E74C3C",       # Red
    "Relu": "#2ECC71",       # Green
    "Sigmoid": "#2ECC71",    # Green
    "Tanh": "#2ECC71",       # Green
    "Add": "#9B59B6",        # Purple
    "Mul": "#9B59B6",        # Purple
    "FusedOp": "#F1C40F"     # Yellow (special highlight)
}


def onnx_to_graph(model_path):
    try:
        model = onnx.load(model_path)
        g = nx.DiGraph()
        for i, node in enumerate(model.graph.node):
            if node.op_type in TRIVIAL_OPS:
                continue  # skip trivial ops for readability

            node_id = node.name if node.name else f"{node.op_type}_{i}"
            label = node.op_type
            g.add_node(node_id, label=label)

            for inp in node.input:
                if not g.has_node(inp):
                    g.add_node(inp, label=inp)
                g.add_edge(inp, node_id)

            for out in node.output:
                if not g.has_node(out):
                    g.add_node(out, label=out)
                g.add_edge(node_id, out)

        print(f"[graph_visualizer] Parsed {len(g.nodes)} nodes, {len(g.edges)} edges from {model_path}")
        return g

    except Exception as e:
        print(f"[graph_visualizer] ERROR parsing {model_path}: {e}")
        return nx.DiGraph()


def dump_graph_json(model_path):
    g = onnx_to_graph(model_path)
    nodes = []
    for n in g.nodes:
        lbl = g.nodes[n].get("label", n)
        color = COLOR_MAP.get(lbl, "#95A5A6")  # grey default
        nodes.append({"id": n, "label": lbl, "color": color})

    edges = [{"source": u, "target": v} for u, v in g.edges]

    if not nodes or not edges:
        print(f"[graph_visualizer] WARNING: Graph from {model_path} is empty.")

    return {"nodes": nodes, "edges": edges}


def simulate_pass_fusion_graph(model_path):
    g = onnx_to_graph(model_path)
    new_nodes = []

    for n in g.nodes:
        lbl = g.nodes[n].get("label", n)
        if "Conv" in lbl or "MatMul" in lbl:
            lbl = "FusedOp"  # mark fused
        color = COLOR_MAP.get(lbl, "#95A5A6")
        new_nodes.append({"id": n, "label": lbl, "color": color})

    edges = [{"source": u, "target": v} for u, v in g.edges]

    print(f"[graph_visualizer] Simulated fusion graph with {len(new_nodes)} nodes, {len(edges)} edges")
    return {"nodes": new_nodes, "edges": edges}
