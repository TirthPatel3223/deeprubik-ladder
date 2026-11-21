"""Value network for a rung: a fully connected residual network on one-hot features.

The architecture follows DeepXube's ``resnet_fc`` heuristic (Forest Agostinelli, MIT
licence, https://github.com/forestagostinelli/deepxube), which is DeepCubeA's network:
one-hot input, a linear layer, residual blocks of two linear layers, and a linear output.

Differences from DeepXube, all for speed on a single GPU:

* the input is a (N, S) uint8 tensor of feature classes that stays on the GPU; the one-hot
  first layer is an embedding sum, which gives the same result as a linear layer applied to
  the one-hot vector without building it;
* the output is multiplied by a fixed ``scale`` so the network learns values of order one
  whatever the action costs are.
"""
from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, dim: int, batch_norm: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.fc1(x)))
        y = self.bn2(self.fc2(y))
        return torch.relu(x + y)


class ResnetFC(nn.Module):
    """Cost-to-go network. ``out_dim`` is 1 for a value network, or the number of actions
    for a deep Q-network with one output per action."""

    def __init__(self, n_slots: int, n_classes: int, res_dim: int = 1000, num_blocks: int = 4,
                 out_dim: int = 1, batch_norm: bool = True, scale: float = 1.0) -> None:
        super().__init__()
        self.n_slots, self.n_classes = n_slots, n_classes
        self.config = dict(n_slots=n_slots, n_classes=n_classes, res_dim=res_dim, num_blocks=num_blocks,
                           out_dim=out_dim, batch_norm=batch_norm, scale=scale)
        self.embed = nn.EmbeddingBag(n_slots * n_classes, res_dim, mode="sum")
        self.bias = nn.Parameter(torch.zeros(res_dim))
        self.bn0 = nn.BatchNorm1d(res_dim) if batch_norm else nn.Identity()
        self.blocks = nn.ModuleList([ResBlock(res_dim, batch_norm) for _ in range(num_blocks)])
        self.out = nn.Linear(res_dim, out_dim)
        self.register_buffer("offsets", torch.arange(n_slots) * n_classes, persistent=False)
        self.scale = scale
        nn.init.normal_(self.embed.weight, std=(1.0 / n_slots) ** 0.5)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: (N, S) integer feature classes -> (N, out_dim) cost-to-go."""
        x = self.embed(states.long() + self.offsets) + self.bias
        x = torch.relu(self.bn0(x))
        for block in self.blocks:
            x = block(x)
        return self.out(x).float() * self.scale


class FusedResnet(nn.Module):
    """Inference copy of a ``ResnetFC`` in eval mode: every batch norm folded into the layer before it, weights in
    ``dtype``. It computes the same function as the network under bf16 autocast (the same rounding of the linear
    layers), with fewer passes over the activations. It is frozen: rebuild it after the network changes."""

    def __init__(self, net: ResnetFC, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()

        def affine(bn: nn.Module, dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
            if isinstance(bn, nn.BatchNorm1d):
                a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                return a, bn.bias - a * bn.running_mean
            return torch.ones(dim, device=device), torch.zeros(dim, device=device)

        dim = net.bias.shape[0]
        dev = net.bias.device
        a0, b0 = affine(net.bn0, dim, dev)
        self.register_buffer("table", (net.embed.weight * a0[None, :]).to(dtype))
        self.register_buffer("bias", (net.bias * a0 + b0).to(dtype))
        self.n_blocks = len(net.blocks)
        for i, blk in enumerate(net.blocks):
            a1, b1 = affine(blk.bn1, dim, dev)
            a2, b2 = affine(blk.bn2, dim, dev)
            for name, t in (("w1", blk.fc1.weight * a1[:, None]), ("c1", blk.fc1.bias * a1 + b1),
                            ("w2", blk.fc2.weight * a2[:, None]), ("c2", blk.fc2.bias * a2 + b2)):
                self.register_buffer(f"{name}_{i}", t.to(dtype))
        self.register_buffer("w_out", net.out.weight.to(dtype))
        self.register_buffer("b_out", net.out.bias.to(dtype))
        self.register_buffer("offsets", net.offsets.clone(), persistent=False)
        self.scale = net.scale

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = nn.functional.embedding_bag(states.long() + self.offsets, self.table, mode="sum")
        x = torch.relu(x + self.bias)
        for i in range(self.n_blocks):
            y = torch.relu(nn.functional.linear(x, getattr(self, f"w1_{i}"), getattr(self, f"c1_{i}")))
            x = torch.relu(x + nn.functional.linear(y, getattr(self, f"w2_{i}"), getattr(self, f"c2_{i}")))
        return nn.functional.linear(x, self.w_out, self.b_out).float() * self.scale


@torch.no_grad()
def fuse(net: nn.Module) -> nn.Module | None:
    """A ``FusedResnet`` of ``net`` if it is a ``ResnetFC`` on a GPU, else None (use the network itself)."""
    if not isinstance(net, ResnetFC) or not net.bias.is_cuda:
        return None
    was_training = net.training
    net.eval()
    fused = FusedResnet(net)
    net.train(was_training)
    return fused.eval()


@torch.no_grad()
def evaluate(net: nn.Module, states: torch.Tensor, chunk: int = 1 << 17, amp: bool = True) -> torch.Tensor:
    """Network output for many states, in chunks, in eval mode. Returns (N, out_dim) float32."""
    was_training = net.training
    net.eval()
    out = []
    use_amp = amp and states.is_cuda
    for s in range(0, states.shape[0], chunk):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out.append(net(states[s:s + chunk]).float())
    net.train(was_training)
    return torch.cat(out) if out else torch.empty((0, 1), device=states.device)
