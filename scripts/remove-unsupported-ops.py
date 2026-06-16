#!/usr/bin/env python3
"""
Replace And and Sign ops with WebGPU-compatible equivalents.
- And(a,b) → Cast→Mul→Cast  (Mul is supported on WebGPU)
- Sign(x)  → Greater/Less/Where chain (all 3 supported on WebGPU)
"""
import os, shutil, time
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto

INPUT = Path("build/onnx_models-dynamo/encoder.onnx")
INPUT_DATA = INPUT.with_suffix(INPUT.suffix + ".data")
BACKUP = True


def _data_name(p):
    return p.with_suffix(p.suffix + ".data").name


def replace_and(graph, node):
    a, b = node.input
    out = node.output[0]
    cast_a = h.make_node("Cast", [a], [f"{out}_cast_a"], to=1, name=f"{node.name}_cast_a")
    cast_b = h.make_node("Cast", [b], [f"{out}_cast_b"], to=1, name=f"{node.name}_cast_b")
    mul = h.make_node("Mul", [f"{out}_cast_a", f"{out}_cast_b"], [f"{out}_mul"],
                      name=f"{node.name}_mul")
    cast_out = h.make_node("Cast", [f"{out}_mul"], [out], to=9, name=f"{node.name}_cast_out")
    idx = list(graph.node).index(node)
    graph.node.remove(node)
    for new_node in [cast_out, mul, cast_b, cast_a]:
        graph.node.insert(idx, new_node)
    for vi_name, vi_type in [(f"{out}_cast_a", TensorProto.FLOAT),
                             (f"{out}_cast_b", TensorProto.FLOAT),
                             (f"{out}_mul", TensorProto.FLOAT)]:
        graph.value_info.append(h.make_tensor_value_info(vi_name, vi_type, None))
    return True


def replace_sign(graph, node):
    x = node.input[0]
    out = node.output[0]
    one = h.make_node("Constant", [], [f"{out}_one"],
                      value=h.make_tensor(f"{out}_one_val", TensorProto.FLOAT, [], [1.0]),
                      name=f"{node.name}_one")
    neg_one = h.make_node("Constant", [], [f"{out}_neg_one"],
                          value=h.make_tensor(f"{out}_neg_one_val", TensorProto.FLOAT, [], [-1.0]),
                          name=f"{node.name}_neg_one")
    zero = h.make_node("Constant", [], [f"{out}_zero"],
                       value=h.make_tensor(f"{out}_zero_val", TensorProto.FLOAT, [], [0.0]),
                       name=f"{node.name}_zero")
    is_pos = h.make_node("Greater", [x, f"{out}_zero"], [f"{out}_is_pos"],
                         name=f"{node.name}_gt")
    is_neg = h.make_node("Less", [x, f"{out}_zero"], [f"{out}_is_neg"],
                         name=f"{node.name}_lt")
    where_neg = h.make_node("Where", [f"{out}_is_neg", f"{out}_neg_one", f"{out}_zero"],
                            [f"{out}_where_neg"], name=f"{node.name}_where_neg")
    where_out = h.make_node("Where", [f"{out}_is_pos", f"{out}_one", f"{out}_where_neg"],
                            [out], name=f"{node.name}_where_out")
    idx = list(graph.node).index(node)
    graph.node.remove(node)
    for new_node in [where_out, where_neg, is_neg, is_pos, zero, neg_one, one]:
        graph.node.insert(idx, new_node)
    for vi_name, vi_type in [(f"{out}_is_pos", TensorProto.BOOL),
                             (f"{out}_is_neg", TensorProto.BOOL),
                             (f"{out}_where_neg", TensorProto.FLOAT)]:
        graph.value_info.append(h.make_tensor_value_info(vi_name, vi_type, None))
    return True


def main():
    print(f"Loading {INPUT} ...")
    model = onnx.load(str(INPUT), load_external_data=True)
    graph = model.graph
    before = len(graph.node)
    print(f"Before: {before} nodes")

    and_nodes = [n for n in graph.node if n.op_type == "And"]
    sign_nodes = [n for n in graph.node if n.op_type == "Sign"]
    print(f"  And  nodes: {len(and_nodes)}")
    print(f"  Sign nodes: {len(sign_nodes)}")
    if not and_nodes and not sign_nodes:
        print("Nothing to replace.")
        return

    t0 = time.time()
    for n in and_nodes:
        replace_and(graph, n)
    for n in sign_nodes:
        replace_sign(graph, n)
    after = len(graph.node)
    print(f"After:  {after} nodes (+{after - before}), {time.time()-t0:.1f}s")

    remaining = {n.op_type for n in graph.node if n.op_type in ("And", "Sign")}
    if remaining:
        print(f"ERROR: still present: {remaining}")
        return
    print("All ops are WebGPU-compatible!")

    # Backup
    if BACKUP:
        backup = INPUT.with_stem(INPUT.stem + ".prewebgpu")
        backup_data = backup.with_suffix(backup.suffix + ".data")
        if not backup.exists():
            shutil.copy2(INPUT, backup)
            if INPUT_DATA.exists():
                shutil.copy2(INPUT_DATA, backup_data)
            print(f"Backup: {backup.name}")

    # Save
    print(f"Saving {INPUT.name} ...")
    if INPUT_DATA.exists():
        INPUT_DATA.unlink()
    data_name = _data_name(INPUT)
    onnx.save_model(model, str(INPUT),
                    save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name)
    mb = (INPUT.stat().st_size + INPUT_DATA.stat().st_size) / (1024 * 1024)
    print(f"Saved ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
