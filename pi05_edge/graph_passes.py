"""Small torch.export graph rewrites needed by one backend or another."""
import torch


def _is_empty_1d(node) -> bool:
    val = node.meta.get("val", None)
    if val is None:
        val = node.meta.get("tensor_meta", None)
    return val is not None and tuple(val.shape) == (0,)


def strip_empty_cats(ep) -> int:
    """Remove `cat([torch.tensor([]), x])` -> `x`.

    transformers' DynamicCache lazily initialises every layer with `torch.tensor([])` and then
    concatenates the first real K/V onto it. PyTorch accepts a rank-1 empty tensor in a rank-4 cat;
    TensorRT does not, so Torch-TensorRT refuses those 72 nodes (18 layers x K/V x two cache
    constructions) and would fall back to PyTorch for them. The result is identical to the non-empty input.
    """
    graph = ep.graph_module.graph
    removed = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.cat.default:
            continue
        tensors = node.args[0]
        if not isinstance(tensors, (list, tuple)):
            continue
        kept = [t for t in tensors if not (hasattr(t, "meta") and _is_empty_1d(t))]
        if len(kept) == 1 and len(kept) < len(tensors):
            node.replace_all_uses_with(kept[0])
            graph.erase_node(node)
            removed += 1
    # the now-unused empty constants (lift_fresh_copy -> detach_) are mutating ops, so DCE keeps them; drop them by hand
    for node in reversed(list(graph.nodes)):
        if node.op == "call_function" and len(node.users) == 0 and node.target in (
            torch.ops.aten.detach_.default, torch.ops.aten.lift_fresh_copy.default, torch.ops.aten.detach.default,
        ):
            graph.erase_node(node)
    graph.eliminate_dead_code()
    ep.graph_module.recompile()
    return removed
