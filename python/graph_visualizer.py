#!/usr/bin/env python3
import onnx
import networkx as nx

# Canonical categories
CANONICAL_MAP = {
    "Conv": "Conv",
    "MatMul": "MatMul/Gemm",
    "Gemm": "MatMul/Gemm",
    "FusedOp": "FusedOp",
    "Add": "Add",
    "Mul": "Mul",
    "Relu": "Activation",
    "Sigmoid": "Activation",
    "Tanh": "Activation",
    "Clip": "Activation",
    "Identity": "Other",
    "Constant": "Other",
    "Flatten": "Other",
    "GlobalAveragePool": "Other",
}

def canonicalize(op_type: str) -> str:
    if op_type in CANONICAL_MAP:
        return CANONICAL_MAP[op_type]
    if op_type.startswith("onnx::Conv"):
        return "Conv"
    return op_type  # fallback: keep original if unknown


def onnx_to_graph(model_path):
    try:
        model = onnx.load(model_path)
        g = nx.DiGraph()
        for i, node in enumerate(model.graph.node):
            op_type = node.op_type
            node_id = node.name if node.name else f"{op_type}_{i}"
            label = canonicalize(op_type)
            g.add_node(node_id, label=label)

            for inp in node.input:
                if not g.has_node(inp):
                    g.add_node(inp, label=canonicalize(inp))
                g.add_edge(inp, node_id)

            for out in node.output:
                if not g.has_node(out):
                    g.add_node(out, label=canonicalize(out))
                g.add_edge(node_id, out)

        print(f"[graph_visualizer] Parsed {len(g.nodes)} nodes, {len(g.edges)} edges from {model_path}")
        return g

    except Exception as e:
        print(f"[graph_visualizer] ERROR parsing {model_path}: {e}")
        return nx.DiGraph()


def dump_graph_json(model_path):
    g = onnx_to_graph(model_path)

    # Collapse into categories for counts
    counts = {}
    for n in g.nodes:
        lbl = g.nodes[n].get("label", n)
        cat = canonicalize(lbl)
        counts[cat] = counts.get(cat, 0) + 1

    nodes = [{"id": n, "label": g.nodes[n].get("label", n)} for n in g.nodes]
    edges = [{"source": u, "target": v} for u, v in g.edges]

    return {"nodes": nodes, "edges": edges, "counts": counts}


def simulate_pass_fusion_graph(model_path):
    g = onnx_to_graph(model_path)
    new_nodes = []

    for n in g.nodes:
        lbl = g.nodes[n].get("label", n)
        if lbl in ("Conv", "MatMul/Gemm"):
            lbl = "FusedOp"
        lbl = canonicalize(lbl)
        new_nodes.append({"id": n, "label": lbl})

    edges = [{"source": u, "target": v} for u, v in g.edges]

    # Collapse into categories for counts
    counts = {}
    for n in new_nodes:
        cat = canonicalize(n["label"])
        counts[cat] = counts.get(cat, 0) + 1

    print(f"[graph_visualizer] Simulated fusion graph with {len(new_nodes)} nodes, {len(edges)} edges")
    return {"nodes": new_nodes, "edges": edges, "counts": counts}
