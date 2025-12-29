# ai_router.py
# Basic POC for operator-level runtime routing using PyTorch FX

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


class RoutedModelWrapper(nn.Module):
    def __init__(self, graph_module, device_map):
        super().__init__()
        self.gm = graph_module
        self.device_map = device_map

    def forward(self, x):
        # Move input to the same device as the first op
        first_dev = list(self.device_map.values())[0]
        x = x.to(first_dev)

        return execute_partitioned(self.gm, self.device_map, x)


# ------------------------
# Operator Router (POC)
# ------------------------
def _preferred_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def route_op(op_name: str, input_shape):
    ops = int(torch.prod(torch.tensor(input_shape))) if input_shape else 1
    accel = _preferred_accelerator()

    if op_name == "linear" and ops > 1_000_000 and accel != "cpu":
        return accel  # accelerator for heavy ops
    elif op_name == "relu":
        return "cpu"  # CPU for light ops
    return accel




# ------------------------
# Runtime Executor
# ------------------------
def execute_partitioned(graph_module, device_map, input_tensor):
    env = {}
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            env[node.name] = input_tensor.to(device_map.get(node.name, 'cpu'))
        elif node.op == 'call_module':
            submod = dict(graph_module.named_modules())[node.target]
            args = [env[arg.name] for arg in node.args]
            dev = device_map.get(node.name, 'cpu')
            submod = submod.to(dev)
            
            # 🛠 FIX: ensure input tensors are on the same device
            args = [a.to(dev) if isinstance(a, torch.Tensor) else a for a in args]

            out = submod(*args)
            env[node.name] = out
            # print(f"[router] → Executing {node.target} on {dev}")

        elif node.op == 'output':
            return env[node.args[0].name]


# ------------------------
# Graph Analyzer + Router
# ------------------------
def optimize(model: nn.Module, batch_size: int = 512, input_dim: int = 2048):
    dummy_input = torch.randn(batch_size, input_dim)  # stay on CPU for tracing
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(dummy_input)

    device_map = {}
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            mod_type = type(dict(gm.named_modules())[node.target]).__name__.lower()
            input_node = node.args[0]
            input_shape = input_node.meta.get('tensor_meta').shape if hasattr(input_node, 'meta') else dummy_input.shape
            assigned = route_op(mod_type, input_shape)
            device_map[node.name] = assigned

    return RoutedModelWrapper(gm, device_map)
