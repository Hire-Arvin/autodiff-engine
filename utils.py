"""Computation graph visualization utilities using Typst/Fletcher."""

from __future__ import annotations
import re
from collections import deque


def to_typst_math(s: str) -> str:
    processed = s
    processed = processed.replace(" @ ", " ")
    processed = re.sub(r"([a-zA-Z])([0-9]+)", r"\1_\2", processed)
    processed = re.sub(r"\.([a-zA-Z]+)", r'."\1"', processed)
    processed = re.sub(r"([a-zA-Z]+)_([a-zA-Z_]+)", r'\1_"\2"', processed)
    processed = re.sub(r"\b(relu|sum)\b", r'"\1"', processed)
    return processed


def _traverse_graph(root):
    nodes, edges = set(), set()
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            nodes.add(v)
            for p in v.parents:
                edges.add((p.parent, v, p.grad.op_str))
                build(p.parent)

    build(root)
    return list(nodes), list(edges)


def _minimize_crossings(nodes_by_level, edges, iterations=4):
    node_to_parents = {n: [] for level in nodes_by_level for n in level}
    node_to_children = {n: [] for level in nodes_by_level for n in level}
    for p, c, _ in edges:
        if p in node_to_children and c in node_to_parents:
            node_to_children[p].append(c)
            node_to_parents[c].append(p)

    ordered_levels = [list(level) for level in nodes_by_level]

    for _ in range(iterations):
        for level_idx in range(1, len(ordered_levels)):
            child_level = ordered_levels[level_idx - 1]
            child_ranks = {node: rank for rank, node in enumerate(child_level)}

            def barycenter_down(node):
                children = node_to_children.get(node, [])
                if not children:
                    return -1
                return sum(child_ranks.get(c, 0) for c in children) / len(children)

            ordered_levels[level_idx].sort(key=barycenter_down)

        for level_idx in range(len(ordered_levels) - 2, -1, -1):
            parent_level = ordered_levels[level_idx + 1]
            parent_ranks = {node: rank for rank, node in enumerate(parent_level)}

            def barycenter_up(node):
                parents = node_to_parents.get(node, [])
                if not parents:
                    return -1
                return sum(parent_ranks.get(p, 0) for p in parents) / len(parents)

            ordered_levels[level_idx].sort(key=barycenter_up)

    return ordered_levels


def _assign_node_levels(root, nodes, edges):
    node_to_id = {node: i for i, node in enumerate(nodes)}
    id_to_node = {i: node for i, node in enumerate(nodes)}

    adj = {node_to_id[n]: [] for n in nodes}
    for p, c, _ in edges:
        if c in node_to_id and p in node_to_id:
            adj[node_to_id[c]].append(node_to_id[p])

    levels = {n: -1 for n in nodes}
    if root in levels:
        levels[root] = 0

    q = deque([root])
    max_level = 0
    visited_bfs = {root}

    while q:
        u = q.popleft()
        u_id = node_to_id[u]
        for v_id in adj.get(u_id, []):
            v = id_to_node.get(v_id)
            if v and v not in visited_bfs:
                levels[v] = levels[u] + 1
                max_level = max(max_level, levels[v])
                visited_bfs.add(v)
                q.append(v)

    nodes_by_level = [[] for _ in range(max_level + 1)]
    for node, level in levels.items():
        if level != -1:
            nodes_by_level[level].append(node)

    for level_nodes in nodes_by_level:
        level_nodes.sort(key=lambda n: n.name)

    nodes_by_level = _minimize_crossings(nodes_by_level, edges)

    node_coords = {}
    for level, nodes_in_level in enumerate(nodes_by_level):
        num_in_level = len(nodes_in_level)
        for rank, node in enumerate(nodes_in_level):
            y_pos = rank - (num_in_level - 1) / 2.0
            node_coords[node] = (-level, y_pos)

    return node_coords


def _generate_typst_source(nodes, edges, coords):
    node_to_id = {node: i for i, node in enumerate(nodes)}

    header = (
        '#import "@preview/fletcher:0.5.8": diagram, node, edge\n'
        "#set page(width: auto, height: auto, margin: 10mm, fill: white)\n"
        '#set text(font: "Linux Libertine", size: 10pt)\n\n'
        "#diagram(cell-size: (40mm, 40mm), {\n"
    )

    node_defs = ""
    for n in nodes:
        if n not in coords:
            continue
        x, y = coords[n]
        nid = node_to_id[n]
        math_name = to_typst_math(n.name)
        shape_str = str(n.value.shape)
        content = (
            f"block(stroke: 0.5pt, inset: 8pt, radius: 4pt, [\n"
            f"#align(center, ${math_name}$)\n"
            f"#line(length: 100%)\n"
            f'#text(size: 9pt, `shape: {shape_str}`)\n])'
        )
        node_defs += f'  node(({x}, {y}), {content}, name: "{nid}")\n'

    edge_defs = ""
    THRESHOLD = 20
    for parent, child, op_str in edges:
        from_id = node_to_id.get(parent)
        to_id = node_to_id.get(child)
        if from_id is None or to_id is None:
            continue
        op_str = op_str or ""
        label = f"${to_typst_math(op_str)}$"
        sep = ", label-sep: 4em" if len(op_str) > THRESHOLD else ""
        edge_defs += f'  edge(label("{from_id}"), label("{to_id}"), "->", label: {label}{sep})\n'

    return header + node_defs + edge_defs + "})"


def draw_graph(root, output_typ_path: str = "computational_graph.typ"):
    """Render the computation graph rooted at `root` to a PDF via Typst."""
    try:
        import typst
    except ImportError:
        raise ImportError("pip install typst to use draw_graph")

    nodes, edges = _traverse_graph(root)
    coords = _assign_node_levels(root, nodes, edges)
    source = _generate_typst_source(nodes, edges, coords)

    with open(output_typ_path, "w", encoding="utf-8") as f:
        f.write(source)
    typst.compile(output_typ_path, output=output_typ_path.replace(".typ", ".pdf"))
