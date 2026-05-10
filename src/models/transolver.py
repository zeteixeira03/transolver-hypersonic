"""
Transolver: a Transformer-based PDE surrogate using Physics-Attention.

Vendored from the official implementation at
https://github.com/thuml/Transolver (MIT License, copyright 2024 THUML @
Tsinghua University). The reference paper is Wu et al., "Transolver: A Fast
Transformer Solver for PDEs on General Geometries", ICML 2024,
arXiv:2402.02366.

Modifications relative to upstream:
- forward takes plain tensors (B, N, C) and an optional position tensor
  instead of a PyTorch Geometric `Data` object, so the module can be tested
  and reused outside the AirfRANS pipeline.
- `unified_pos` reference grid bounds and resolution are parameters, not
  hard-coded to the AirfRANS spatial domain.
- snake_case for the public wrapper module's function-style helpers; the
  PyTorch Module classes keep CamelCase as is conventional.

Equation numbers refer to arXiv:2402.02366.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


# ============================================================================================
#                                       activations
# ============================================================================================
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


# ============================================================================================
#                                       physics-attention block
# ============================================================================================
class PhysicsAttentionIrregularMesh(nn.Module):
    """
    Physics-Attention for point clouds / irregular meshes.

    The block (Eqs. 6-9 in arXiv:2402.02366) performs three steps:

    1. Slice. Each of N points is softly assigned to one of M learned slices
       via softmax weights w(x_i, j). A slice token z_j is the
       weight-averaged feature aggregated from all points belonging to that
       slice.
    2. Self-attention among the M slice tokens. This is the O(M^2) part that
       replaces standard O(N^2) point-wise attention.
    3. Deslice. Each point's output is the weighted sum of the attended slice
       tokens, weighted by the same slice assignments from step 1.

    The slicing is per-head: heads can specialize on different physical
    regimes (e.g. boundary layer vs free stream).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # learnable softmax temperature, broadcast over (B, H, N, M)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        # projections for the slice step: x_mid drives slice assignments,
        # fx_mid is the per-point feature being aggregated into slice tokens
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        # orthogonal init on the slice-assignment projection encourages
        # diverse slice prototypes at initialization
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        b, n, _ = x.shape

        # step 1: slice. project to (B, H, N, dim_head), then compute
        # slice-assignment logits in (B, H, N, M) and softmax over M.
        fx_mid = (
            self.in_project_fx(x)
            .reshape(b, n, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(b, n, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        # temperature is learnable but unbounded upstream; clamp magnitude to
        # avoid division by ~0 producing inf logits under fp16 autocast.
        temp = self.temperature.abs().clamp_min(0.1)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / temp)  # (B, H, N, M)
        slice_norm = slice_weights.sum(2)  # (B, H, M), sum over points per slice
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5)[:, :, :, None].repeat(
            1, 1, 1, self.dim_head
        )

        # step 2: self-attention among the M slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attn, v)  # (B, H, M, dim_head)

        # step 3: deslice. broadcast attended slice tokens back to points
        # using the same slice-assignment weights.
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


# ============================================================================================
#                                       feed-forward MLP
# ============================================================================================
class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ) -> None:
        super().__init__()
        if act not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {act}")
        act_cls = ACTIVATIONS[act]
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_cls())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_cls()) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


# ============================================================================================
#                                       transolver block
# ============================================================================================
class TransolverBlock(nn.Module):
    """Pre-norm Transformer encoder block with Physics-Attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregularMesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.head(self.ln_3(fx))
        return fx


# ============================================================================================
#                                       full transolver model
# ============================================================================================
class Transolver(nn.Module):
    """
    Stack of Transolver blocks for point-cloud regression.

    Inputs are a per-point feature tensor of shape (B, N, space_dim + fun_dim)
    where the spatial coordinates occupy the first `space_dim` channels and
    optional auxiliary functions occupy the remaining `fun_dim` channels.
    Outputs are per-point predictions of shape (B, N, out_dim).

    `unified_pos` adds a fixed reference-grid distance encoding of size
    `ref**2` to each point, useful when comparing very different geometries
    in the same coordinate system. The grid is parameterized by its bounds
    and resolution rather than hard-coded.

    Parameters
    ----------
    space_dim : int
        Number of spatial coordinate channels in the input.
    fun_dim : int
        Number of additional input function channels.
    out_dim : int
        Number of output channels per point.
    n_hidden : int
        Hidden dimension throughout the encoder.
    n_layers : int
        Number of TransolverBlocks.
    n_head : int
        Number of attention heads.
    mlp_ratio : int
        Hidden expansion factor inside each block's MLP.
    slice_num : int
        Number of physics slices M used by Physics-Attention.
    dropout : float
        Dropout applied inside attention and the output projection.
    act : str
        Activation name; one of ACTIVATIONS.
    unified_pos : bool
        Whether to append reference-grid distance features.
    grid_ref : int
        Resolution of the reference grid along each axis.
    grid_bounds : tuple of float
        Reference-grid bounds (xmin, xmax, ymin, ymax). 2D only for now.
    """

    def __init__(
        self,
        space_dim: int = 2,
        fun_dim: int = 0,
        out_dim: int = 1,
        n_hidden: int = 128,
        n_layers: int = 4,
        n_head: int = 8,
        mlp_ratio: int = 2,
        slice_num: int = 32,
        dropout: float = 0.0,
        act: str = "gelu",
        unified_pos: bool = False,
        grid_ref: int = 8,
        grid_bounds: tuple[float, float, float, float] = (-2.0, 4.0, -1.5, 1.5),
    ) -> None:
        super().__init__()
        self.space_dim = space_dim
        self.fun_dim = fun_dim
        self.unified_pos = unified_pos
        self.grid_ref = grid_ref
        self.grid_bounds = grid_bounds

        if unified_pos:
            in_dim = space_dim + fun_dim + grid_ref * grid_ref
        else:
            in_dim = space_dim + fun_dim
        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden))
        self._initialize_weights()

    # --------------------------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        def init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init)

    # --------------------------------------------------------------------------------------
    def _build_grid_features(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute per-point distances to a fixed reference grid.

        Parameters
        ----------
        pos : torch.Tensor
            Point coordinates of shape (B, N, 2).

        Returns
        -------
        torch.Tensor
            Distance features of shape (B, N, grid_ref**2).
        """
        b = pos.shape[0]
        xmin, xmax, ymin, ymax = self.grid_bounds
        gx = torch.tensor(np.linspace(xmin, xmax, self.grid_ref), dtype=torch.float32, device=pos.device)
        gy = torch.tensor(np.linspace(ymin, ymax, self.grid_ref), dtype=torch.float32, device=pos.device)
        gx = gx.reshape(1, self.grid_ref, 1, 1).expand(b, -1, self.grid_ref, 1)
        gy = gy.reshape(1, 1, self.grid_ref, 1).expand(b, self.grid_ref, -1, 1)
        grid = torch.cat((gx, gy), dim=-1).reshape(b, self.grid_ref ** 2, 2)
        diff = pos[:, :, None, :] - grid[:, None, :, :]
        return torch.sqrt((diff ** 2).sum(dim=-1))

    # --------------------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Per-point input features of shape (B, N, space_dim + fun_dim).
            The first `space_dim` channels are taken to be coordinates.
        pos : torch.Tensor, optional
            Point coordinates of shape (B, N, 2), required when
            `unified_pos` is True. If None, `x[..., :2]` is used.

        Returns
        -------
        torch.Tensor
            Per-point predictions of shape (B, N, out_dim).
        """
        if self.unified_pos:
            if pos is None:
                pos = x[..., :2]
            x = torch.cat((x, self._build_grid_features(pos)), dim=-1)

        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx
