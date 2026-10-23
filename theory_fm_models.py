"""Theory-guided flow-field definitions, v1.0.

MODEL DEFINITIONS ONLY: no trainer, state integrator, downloaded data, test suite,
random Brownian increments, recurrent network, or hidden history cache.

Source: user manuscript, 'Complex-Order Field-Memory McKean--Vlasov Dynamics',
2026-10-07, especially (2.1)--(2.11), (3.13), (7.9), and (8.2).

Five MAIN model IDs (new IDs, NOT aliases for the old M0--M3 experiments):
    FM_STANDARD           ordinary time-conditioned MLP velocity, ODE
    FM_REAL_ODE           shared-potential real-order field memory, ODE
    FM_COMPLEX_SDE        shared-potential complex-order field memory, SDE
    FM_COMPLEX_ODE        identical complex drift architecture, ODE
    FM_COMPLEX_VDC_ODE    two-potential compatible div/curl construction, ODE

All neural networks are stateless tanh MLPs. Analytic temporal memory is explicit
in FieldHistory: it acts on F[rho_s](x) at the SAME query x for all s. It NEVER
acts on the receiving particle's past positions or on data-observation time.

D=2: J maps (u,v) to (-v,u), matching multiplication by i in the manuscript.
Even D>2: fixed orthogonal complex structure J; this is an explicitly defined
engineering extension, NOT a high-dimensional theorem from the manuscript.
The VDC extension uses two probability potentials and is NOT the original
single-functional model. Higher-dimensional curl is an antisymmetric tensor.

Temporal numerical representations (both keep the RL initial contribution):
  node: exact RL derivative of the piecewise-LINEAR field history, Re(alpha)<1.
  cell_average: exact cell-averaged RL derivative of piecewise-CONSTANT field
      history, useful for the first interval where point drift can be singular.
These are field approximations, not a state solver or a convergence theorem.

Use float32 or float64, not half/autocast around spatial differentiation. All
complex kernel calculations use float64/complex128 and are differentiable.
Definitions have undergone static syntax checking only in the delivery session.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple
import math

import torch
from torch import Tensor, nn

__version__ = "1.0.0"

MAIN_MODELS = (
    "FM_STANDARD", "FM_REAL_ODE", "FM_COMPLEX_SDE", "FM_COMPLEX_ODE",
    "FM_COMPLEX_VDC_ODE",
)
CONTROL_MODELS = (
    "FM_STANDARD_SDE", "FM_FIELD_ZERO_ODE", "FM_FIELD_ZERO_SDE",
    "FM_REAL_SDE", "FM_REAL_ROT_ODE", "FM_VDC_REAL_ODE",
    "FM_VDC_ZERO_ODE", "FM_COMPLEX_VDC_SDE", "FM_COMPLEX_FIXED_ODE",
    "FM_FIELD_FIR_ODE",
)


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    condition_dim: int = 0
    hidden_dim: int = 128
    hidden_layers: int = 3
    time_frequencies: int = 4
    # These are numerical experiment bounds, NOT the domain of complex calculus.
    order_min: float = -1.0
    order_max: float = 0.40
    imag_bound: float = 0.50
    order_real_init: Tuple[float, ...] = (-0.60, -0.20, 0.0, 0.20)
    order_imag_init: Tuple[float, ...] = (0.15, -0.10, 0.0, 0.10)
    reference_time: float = 1.0
    coefficient_bound: float = 2.0
    diffusion_kappa: float = 0.05
    potential_amplitude: float = 8.0
    # Actual Gaussian radius = radius_per_coordinate * sqrt(D).
    external_radius_per_coordinate: float = 4.0
    interaction_radius_per_coordinate: float = 2.0
    source_chunk: int = 32
    query_chunk: int = 16
    fir_length: int = 33
    fir_dt: float = 1.0 / 32.0

    def __post_init__(self) -> None:
        if self.state_dim < 2 or self.state_dim % 2:
            raise ValueError("state_dim must be even and >=2. Do not silently pad or drop a coordinate.")
        if self.condition_dim < 0 or self.hidden_dim < 1 or self.hidden_layers < 1:
            raise ValueError("Invalid feedforward dimensions.")
        if not self.order_min < 0 < self.order_max < 0.5:
            raise ValueError("Default experiment family requires order_min<0<order_max<1/2.")
        if self.imag_bound <= 0 or self.reference_time <= 0:
            raise ValueError("imag_bound and reference_time must be positive.")
        if len(self.order_real_init) == 0 or len(self.order_real_init) != len(self.order_imag_init):
            raise ValueError("Initial real and imaginary orders must have the same nonzero length.")
        if any(not self.order_min < a < self.order_max for a in self.order_real_init):
            raise ValueError("Real initial orders must lie strictly inside the configured range.")
        if any(abs(b) >= self.imag_bound for b in self.order_imag_init):
            raise ValueError("Imaginary initial orders must lie strictly inside the configured range.")
        if self.coefficient_bound <= 1.0 or self.diffusion_kappa <= 0:
            raise ValueError("coefficient_bound>1 and diffusion_kappa>0 are required.")
        if min(self.potential_amplitude, self.external_radius_per_coordinate,
               self.interaction_radius_per_coordinate) <= 0:
            raise ValueError("Potential scales must be positive.")
        if min(self.source_chunk, self.query_chunk, self.fir_length) < 1 or self.fir_dt <= 0:
            raise ValueError("Invalid chunk or FIR configuration.")


@dataclass(frozen=True)
class FieldHistory:
    """Explicit empirical probability history; no neural hidden state.

    times:     [H], t[0]=0, strictly increasing; irregular grids are supported.
    particles: [E,H,N,D]. E=independent ensembles/conditions; N=particles of ONE
               conditional law, not N unrelated robot conditions.
    weights:   optional [E,H,N], nonnegative and summing to one at each snapshot.

    A stored past source particle y_s is allowed. The receiving query stays at
    the same x while the whole field history is evaluated. Do not put computed
    velocities along a moving receiving trajectory into this container.
    """
    times: Tensor
    particles: Tensor
    weights: Optional[Tensor] = None

    def validate(self, state_dim: int) -> None:
        _real_tensor(self.times, "history.times")
        _real_tensor(self.particles, "history.particles")
        if self.times.ndim != 1 or self.particles.ndim != 4:
            raise ValueError("Expected times[H], particles[E,H,N,D].")
        e, h, n, d = self.particles.shape
        if min(e, h, n) < 1 or d != state_dim or self.times.numel() != h:
            raise ValueError("History dimensions disagree.")
        if self.times.device != self.particles.device:
            raise ValueError("Times and particles must be on the same device.")
        if float(self.times[0].detach()) != 0.0:
            raise ValueError("History must start at zero; never reset history at a solver restart.")
        if h > 1 and not bool(torch.all(self.times[1:] > self.times[:-1])):
            raise ValueError("History times must be strictly increasing.")
        if not bool(torch.isfinite(self.times).all()):
            raise ValueError("Non-finite history times.")
        if self.weights is not None:
            _real_tensor(self.weights, "history.weights")
            if self.weights.shape != (e, h, n) or self.weights.device != self.particles.device:
                raise ValueError("Expected probability weights[E,H,N] on the same device.")
            if not bool(torch.isfinite(self.weights).all()) or bool((self.weights < 0).any()):
                raise ValueError("Weights must be finite and nonnegative.")
            if not torch.allclose(self.weights.sum(-1), torch.ones_like(self.weights.sum(-1)),
                                  atol=2e-5, rtol=2e-5):
                raise ValueError("Each snapshot's probability weights must sum to one.")

    def prefix(self, count: int) -> "FieldHistory":
        if not 1 <= count <= self.times.numel():
            raise ValueError("Invalid history prefix length.")
        return FieldHistory(self.times[:count], self.particles[:, :count],
                            None if self.weights is None else self.weights[:, :count])

    def detach(self) -> "FieldHistory":
        """Explicit caller choice for teacher histories, NEVER done silently in a model."""
        return FieldHistory(self.times.detach(), self.particles.detach(),
                            None if self.weights is None else self.weights.detach())


@dataclass
class FieldOutput:
    drift: Tensor                  # [E,Q,D]; for ODE this is the transport velocity
    diffusion: float               # scalar sqrt(2*kappa), NOT a noise sample
    model_id: str
    representation: str            # "node_pl" or "cell_average_pc" (standard uses quadrature)
    divergence: Optional[Tensor] = None  # [E,Q]
    curl: Optional[Tensor] = None        # [E,Q,P], Omega_ij=partial_i b_j-partial_j b_i
    curl_pairs: Tuple[Tuple[int, int], ...] = ()
    divergence_estimator: str = "not_computed"

    @property
    def velocity(self) -> Tensor:
        """Alias; for an SDE it is drift, NOT probability-current velocity j/rho."""
        return self.drift


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: Literal["direct", "shared_potential", "two_potential"]
    temporal: Literal["none", "zero", "real", "complex", "real_rot", "complex_fixed", "fir"]
    stochastic: bool


SPECS: Dict[str, ModelSpec] = {
    "FM_STANDARD": ModelSpec("FM_STANDARD", "direct", "none", False),
    "FM_REAL_ODE": ModelSpec("FM_REAL_ODE", "shared_potential", "real", False),
    "FM_COMPLEX_SDE": ModelSpec("FM_COMPLEX_SDE", "shared_potential", "complex", True),
    "FM_COMPLEX_ODE": ModelSpec("FM_COMPLEX_ODE", "shared_potential", "complex", False),
    "FM_COMPLEX_VDC_ODE": ModelSpec("FM_COMPLEX_VDC_ODE", "two_potential", "complex", False),
    "FM_STANDARD_SDE": ModelSpec("FM_STANDARD_SDE", "direct", "none", True),
    "FM_FIELD_ZERO_ODE": ModelSpec("FM_FIELD_ZERO_ODE", "shared_potential", "zero", False),
    "FM_FIELD_ZERO_SDE": ModelSpec("FM_FIELD_ZERO_SDE", "shared_potential", "zero", True),
    "FM_REAL_SDE": ModelSpec("FM_REAL_SDE", "shared_potential", "real", True),
    "FM_REAL_ROT_ODE": ModelSpec("FM_REAL_ROT_ODE", "shared_potential", "real_rot", False),
    "FM_VDC_REAL_ODE": ModelSpec("FM_VDC_REAL_ODE", "two_potential", "real", False),
    "FM_VDC_ZERO_ODE": ModelSpec("FM_VDC_ZERO_ODE", "two_potential", "zero", False),
    "FM_COMPLEX_VDC_SDE": ModelSpec("FM_COMPLEX_VDC_SDE", "two_potential", "complex", True),
    "FM_COMPLEX_FIXED_ODE": ModelSpec("FM_COMPLEX_FIXED_ODE", "shared_potential", "complex_fixed", False),
    "FM_FIELD_FIR_ODE": ModelSpec("FM_FIELD_FIR_ODE", "shared_potential", "fir", False),
}


def _real_tensor(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or x.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a real float32/float64 tensor; half and complex states are unsupported.")


def _condition(c: Optional[Tensor], e: int, d: int, like: Tensor) -> Tensor:
    if d == 0:
        if c is not None and c.shape != (e, 0):
            raise ValueError("This model was configured as unconditional.")
        return like.new_empty(e, 0)
    if c is None or c.shape != (e, d):
        raise ValueError(f"Expected one condition[E,{d}] per ensemble, not one unrelated condition per particle.")
    _real_tensor(c, "condition")
    if c.device != like.device or c.dtype != like.dtype:
        raise ValueError("Condition, model and queries must share dtype and device.")
    return c


def _broadcast_condition(c: Tensor, x: Tensor) -> Tensor:
    shape = (c.shape[0],) + (1,) * (x.ndim - 2) + (c.shape[-1],)
    return c.reshape(shape).expand(*x.shape[:-1], c.shape[-1])


def _time_like(t: Tensor | float, x: Tensor) -> Tensor:
    z = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if z.ndim == 0:
        return z.expand(x.shape[0], x.shape[1])
    if z.shape == (x.shape[0],):
        return z[:, None].expand(x.shape[0], x.shape[1])
    if z.shape == x.shape[:-1]:
        return z
    raise ValueError("Time must be a scalar, [E], or [E,Q].")


def _same_time(t: Tensor | float, scalar: Tensor) -> bool:
    z = torch.as_tensor(t, device=scalar.device, dtype=scalar.dtype)
    return bool(torch.allclose(z, scalar.expand_as(z), atol=2e-7, rtol=2e-6))


def canonical_complex_structure(d: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if d < 2 or d % 2:
        raise ValueError("An orthogonal complex structure requires an even real dimension.")
    j = torch.zeros(d, d, dtype=dtype)
    for i in range(0, d, 2):
        j[i, i + 1], j[i + 1, i] = -1.0, 1.0
    return j


def _check_complex_structure(j: Tensor, d: int) -> None:
    _real_tensor(j, "J")
    if j.shape != (d, d):
        raise ValueError("J must have shape [D,D].")
    jd = j.detach().double()
    eye = torch.eye(d, dtype=torch.float64, device=jd.device)
    if not torch.allclose(jd + jd.T, torch.zeros_like(jd), atol=2e-5, rtol=2e-5):
        raise ValueError("J must be skew-symmetric.")
    if not torch.allclose(jd @ jd, -eye, atol=2e-5, rtol=2e-5):
        raise ValueError("J must satisfy J^2=-I. Keep the same J in all related models.")


def apply_j(v: Tensor, j: Tensor) -> Tensor:
    return v @ j.T


class FeedForward(nn.Module):
    """Identical primitive in every model: Linear/Tanh stacks, no recurrent state."""
    def __init__(self, input_dim: int, output_dim: int, width: int, layers: int) -> None:
        super().__init__()
        modules = []
        n = input_dim
        for _ in range(layers):
            modules.extend([nn.Linear(n, width), nn.Tanh()])
            n = width
        modules.append(nn.Linear(n, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SmoothProbabilityPotentials(nn.Module):
    """U_l[mu](x,c)=V_l(x,c)+integral W_l(x-y,c) mu(dy).

    V/W are separate stateless MLPs. Gaussian envelopes make V smooth bounded
    and W rapidly decaying for every finite parameter vector and fixed c.
    W is even by construction. channels=1 recovers the shared-potential family;
    channels=2 defines the NEW two-functional Hodge ansatz, not the paper theorem.
    The potentials do not take generation time as an input.
    """
    def __init__(self, cfg: ModelConfig, channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.channels = channels
        n = cfg.state_dim + cfg.condition_dim
        self.v_net = FeedForward(n, channels, cfg.hidden_dim, cfg.hidden_layers)
        self.w_net = FeedForward(n, channels, cfg.hidden_dim, cfg.hidden_layers)
        self.register_buffer("v_radius", torch.tensor(cfg.external_radius_per_coordinate * math.sqrt(cfg.state_dim)))
        self.register_buffer("w_radius", torch.tensor(cfg.interaction_radius_per_coordinate * math.sqrt(cfg.state_dim)))

    def _potential(self, z: Tensor, c: Tensor, *, interaction: bool) -> Tensor:
        cc = _broadcast_condition(c, z)
        net = self.w_net if interaction else self.v_net
        zc = torch.cat((z, cc), dim=-1)
        logits = net(zc)
        if interaction:
            logits = 0.5 * (logits + net(torch.cat((-z, cc), dim=-1)))
        radius = self.w_radius if interaction else self.v_radius
        envelope = torch.exp(-0.5 * (z / radius).square().sum(-1, keepdim=True))
        return self.cfg.potential_amplitude * torch.tanh(logits) * envelope

    def values(self, x: Tensor, sources: Tensor, weights: Optional[Tensor], c: Tensor) -> Tensor:
        # x[E,Q,D], sources[E,N,D], output[E,Q,channels].
        # All receiving query coordinates remain fixed throughout this function.
        value = self._potential(x, c, interaction=False)
        n = sources.shape[1]
        for left in range(0, n, self.cfg.source_chunk):
            right = min(left + self.cfg.source_chunk, n)
            delta = x[:, :, None, :] - sources[:, None, left:right, :]
            w = self._potential(delta, c, interaction=True)
            if weights is None:
                value = value + w.sum(dim=2) / n
            else:
                value = value + (w * weights[:, None, left:right, None]).sum(dim=2)
        return value


# Differentiable Lanczos formula, only in Re(z)>1/2 (all arguments here satisfy it).
_LANCZOS = (
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
)


def complex_loggamma_right(z: Tensor) -> Tensor:
    if not z.is_complex():
        z = torch.complex(z.double(), torch.zeros_like(z, dtype=torch.float64))
    z = z.to(torch.complex128)
    if bool((z.real.detach() <= 0.5).any()):
        raise ValueError("Lanczos helper is deliberately limited to Re(z)>1/2.")
    zm = z - 1.0
    series = torch.full_like(z, _LANCZOS[0])
    for k, p in enumerate(_LANCZOS[1:], start=1):
        series = series + p / (zm + k)
    t = zm + 7.5
    return 0.5 * math.log(2.0 * math.pi) + (zm + 0.5) * torch.log(t) - t + torch.log(series)


def positive_real_complex_power(r: Tensor, z: Tensor) -> Tensor:
    """r[...,H]>=0, z[K]; output[...,H,K]. 0^z=0 only for Re(z)>0.

    No log(0), no branch-changing epsilon. Positive arguments use their real log.
    """
    rr = r.to(torch.float64)
    zz = z.to(torch.complex128)
    if bool((rr.detach() < 0).any()) or bool((zz.real.detach() <= 0).any()):
        raise ValueError("This helper needs r>=0 and Re(exponent)>0.")
    safe_r = torch.where(rr > 0, rr, torch.ones_like(rr))
    powered = torch.exp(torch.log(safe_r)[..., None] * zz)
    return torch.where((rr > 0)[..., None], powered, torch.zeros_like(powered))


class RLFieldSpectrum(nn.Module):
    """Fixed-during-transport, learnable-between-updates complex order spectrum.

    Main family uses REAL signed coefficients, so complex-order benefits are not
    confused with freely chosen complex zero-order coefficients. real_rot is a
    separate control. Parameters are stored real and converted without .item()
    or .detach(), preserving order gradients.
    """
    def __init__(self, cfg: ModelConfig, mode: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        a = torch.tensor(cfg.order_real_init, dtype=torch.float64)
        ratio = (a - cfg.order_min) / (cfg.order_max - cfg.order_min)
        raw_a = torch.logit(ratio)
        learn_a = mode in ("real", "complex", "real_rot")
        self.raw_a = nn.Parameter(raw_a, requires_grad=learn_a)
        if mode in ("complex", "complex_fixed"):
            b = torch.tensor(cfg.order_imag_init, dtype=torch.float64)
            self.raw_b = nn.Parameter(torch.atanh(b / cfg.imag_bound), requires_grad=(mode == "complex"))
        else:
            self.register_buffer("raw_b", torch.zeros_like(a))
        # c_k initially 1/K, sum of coefficients initially one; no forced positivity.
        self.raw_c = nn.Parameter(torch.full_like(a, math.atanh(1.0 / cfg.coefficient_bound)))
        if mode == "real_rot":
            self.raw_ci = nn.Parameter(torch.zeros_like(a))
        else:
            self.register_buffer("raw_ci", torch.zeros_like(a))
        if mode == "zero":
            # Exact identity; no unused trainable spectrum parameters.
            self.raw_a.requires_grad_(False)
            self.raw_c.requires_grad_(False)

    def parameters_in_order_space(self) -> Tuple[Tensor, Tensor]:
        a = self.cfg.order_min + (self.cfg.order_max - self.cfg.order_min) * torch.sigmoid(self.raw_a.double())
        if self.mode in ("complex", "complex_fixed"):
            b = self.cfg.imag_bound * torch.tanh(self.raw_b.double())
        else:
            b = torch.zeros_like(a)
        n = a.numel()
        cr = self.cfg.coefficient_bound * torch.tanh(self.raw_c.double()) / n
        ci = self.cfg.coefficient_bound * torch.tanh(self.raw_ci.double()) / n if self.mode == "real_rot" else torch.zeros_like(a)
        return torch.complex(a, b), torch.complex(cr, ci)

    def weights(self, times: Tensor, *, next_time: Optional[Tensor | float] = None) -> Tensor:
        """Complex weights[H] on fields evaluated at ONE fixed spatial location.

        next_time=None: node drift at t[-1] of the PL interpolant (requires H>=2).
        next_time>t[-1]: averaged drift over [t[-1],next_time] of the PC history.
        """
        h = times.numel()
        if self.mode == "zero":
            one = torch.ones(1, dtype=torch.complex128, device=times.device)
            return torch.cat((torch.zeros(h - 1, dtype=torch.complex128, device=times.device), one))
        alpha, coeff = self.parameters_in_order_space()
        tt = times.to(torch.float64)
        scale = torch.exp(alpha * math.log(self.cfg.reference_time)) * coeff
        p = 1.0 - alpha
        inv_gamma2 = torch.exp(-complex_loggamma_right(2.0 - alpha))
        if next_time is None:
            if h < 2 or not bool(tt[-1] > 0):
                raise ValueError("A positive-order RL point drift is generally singular at t=0. Use cell_average.")
            t = tt[-1]
            # (f_{j+1}-f_j)/(t_{j+1}-t_j) integrated against (t-s)^(-alpha).
            av = (positive_real_complex_power(t - tt[:-1], p)
                  - positive_real_complex_power(t - tt[1:], p))
            av = av * inv_gamma2 / (tt[1:] - tt[:-1])[:, None]
            # gamma(2-alpha)=(1-alpha)*gamma(1-alpha).
            initial = torch.exp(-alpha * torch.log(t)) * (1.0 - alpha) * inv_gamma2
            zero = torch.zeros(1, alpha.numel(), dtype=torch.complex128, device=times.device)
            weights = torch.cat((zero, av), dim=0) - torch.cat((av, zero), dim=0)
            e0 = torch.cat((torch.ones(1, dtype=torch.float64, device=times.device),
                            torch.zeros(h - 1, dtype=torch.float64, device=times.device)))
            weights = weights + e0[:, None] * initial
        else:
            end = torch.as_tensor(next_time, dtype=torch.float64, device=times.device)
            if end.ndim != 0 or not bool(end > tt[-1]):
                raise ValueError("next_time must be a scalar strictly after the final history snapshot.")
            left = tt[-1]
            starts = tt
            ends = torch.cat((tt[1:], end.reshape(1)))
            # Integral of D^alpha f over the final cell = I^(1-alpha) f(end)-I^(1-alpha) f(left).
            at_end = (positive_real_complex_power(end - starts, p)
                      - positive_real_complex_power(end - ends, p))
            at_left = (positive_real_complex_power(torch.clamp(left - starts, min=0), p)
                       - positive_real_complex_power(torch.clamp(left - ends, min=0), p))
            weights = (at_end - at_left) * inv_gamma2 / (end - left)
        return (weights * scale[None, :]).sum(dim=-1)


class FIRFieldMemory(nn.Module):
    """Free real lag weights on the SAME Eulerian field snapshots, not on states.

    Fixed lag spacing. Used only as a finite-grid control, with no claim of a
    continuous-order or grid-invariant meaning. At startup older values are zero.
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        w = torch.zeros(cfg.fir_length, dtype=torch.float64)
        w[0] = 1.0
        self.lag_weights = nn.Parameter(w)

    def weights(self, times: Tensor, *, next_time: Optional[Tensor | float] = None) -> Tensor:
        if times.numel() > 1:
            expected = torch.full_like(times[1:], self.cfg.fir_dt)
            if not torch.allclose(times[1:] - times[:-1], expected, atol=2e-7, rtol=2e-5):
                raise ValueError("FIR requires its declared fixed lag spacing. Do not silently interpolate taps.")
        if next_time is not None and abs(float(torch.as_tensor(next_time).detach())
                                         - float(times[-1].detach()) - self.cfg.fir_dt) > 2e-6:
            raise ValueError("FIR interval must equal fir_dt.")
        h = times.numel()
        if h > self.cfg.fir_length:
            raise ValueError("FIR horizon exceeded; increase the predeclared kernel length or use a separate truncated-history ablation.")
        w = self.lag_weights[:h].flip(0)
        return torch.complex(w, torch.zeros_like(w))


class FlowFieldModel(nn.Module):
    """Unified stateless interface for the five requested models and controls."""
    def __init__(self, spec: ModelSpec, cfg: ModelConfig, j: Optional[Tensor] = None) -> None:
        super().__init__()
        self.spec = spec
        self.cfg = cfg
        self.kappa = cfg.diffusion_kappa if spec.stochastic else 0.0
        jj = canonical_complex_structure(cfg.state_dim) if j is None else j.detach().clone()
        _check_complex_structure(jj, cfg.state_dim)
        self.register_buffer("J", jj)
        if spec.family == "direct":
            time_dim = 1 + 2 * cfg.time_frequencies
            self.direct = FeedForward(cfg.state_dim + cfg.condition_dim + time_dim,
                                      cfg.state_dim, cfg.hidden_dim, cfg.hidden_layers)
            self.potentials = None
            self.temporal = None
        else:
            self.direct = None
            self.potentials = SmoothProbabilityPotentials(cfg, 2 if spec.family == "two_potential" else 1)
            self.temporal = (FIRFieldMemory(cfg) if spec.temporal == "fir"
                             else RLFieldSpectrum(cfg, spec.temporal))

    @property
    def needs_history(self) -> bool:
        return self.spec.family != "direct"

    @property
    def diffusion(self) -> float:
        return math.sqrt(2.0 * self.kappa)

    def _validate_x(self, x: Tensor, history: Optional[FieldHistory]) -> None:
        _real_tensor(x, "queries")
        if x.ndim != 3 or x.shape[-1] != self.cfg.state_dim or min(x.shape[:2]) < 1:
            raise ValueError("Expected queries[E,Q,D]. Flatten the sample, not the batch or ensemble.")
        parameter = next(self.direct.parameters()) if self.direct is not None else next(self.potentials.parameters())
        if parameter.dtype != x.dtype or parameter.device != x.device:
            raise ValueError("Queries and network parameters must share dtype/device; use model.to(...).")
        if history is not None:
            history.validate(self.cfg.state_dim)
            if history.particles.shape[0] != x.shape[0]:
                raise ValueError("Queries and history must have the same number of independent ensembles.")
            if history.particles.device != x.device or history.particles.dtype != x.dtype:
                raise ValueError("Queries and reference particles must share dtype/device.")
        if self.needs_history and history is None:
            raise ValueError("This model requires an explicit law history; no hidden cache is constructed.")

    def _direct_velocity(self, x: Tensor, t: Tensor | float, c: Tensor) -> Tensor:
        tt = _time_like(t, x)
        features = [tt[..., None]]
        for k in range(self.cfg.time_frequencies):
            phase = (2.0 ** k) * math.pi * tt
            features.extend((phase.sin()[..., None], phase.cos()[..., None]))
        inp = torch.cat((x, _broadcast_condition(c, x), *features), dim=-1)
        return self.direct(inp)

    def _field_chunk(self, x: Tensor, history: FieldHistory, c: Tensor,
                     weights: Tensor, *, create_graph: bool) -> Tensor:
        # Cloning the receiving query creates a DIFFERENT graph input from the
        # source particles, even if both originated from the same ensemble tensor.
        # Partial_x therefore holds the empirical law fixed, without silently
        # detaching its parameter/history dependencies in the outer graph.
        out = None
        for j in range(history.times.numel()):
            xq = x.clone()
            if not xq.requires_grad:
                xq.requires_grad_(True)
            ww = None if history.weights is None else history.weights[:, j]
            values = self.potentials.values(xq, history.particles[:, j], ww, c)
            forces = []
            for ch in range(self.potentials.channels):
                grad = torch.autograd.grad(values[..., ch].sum(), xq, create_graph=create_graph,
                                           retain_graph=(create_graph or ch + 1 < self.potentials.channels))[0]
                forces.append(-grad)
            # Single U: F=-grad U. Two U/S: F=-grad U-J grad S.
            base = forces[0]
            if len(forces) == 2:
                base = base + apply_j(forces[1], self.J)
            q = weights[j].real.to(dtype=x.dtype)
            r = weights[j].imag.to(dtype=x.dtype)
            term = q * base + r * apply_j(base, self.J)
            out = term if out is None else out + term
        return out

    def forward(self, queries: Tensor, t: Tensor | float,
                history: Optional[FieldHistory] = None, condition: Optional[Tensor] = None,
                *, create_graph: Optional[bool] = None) -> FieldOutput:
        """Point drift at positive t; source-field history uses PL reconstruction.

        For all particles in one call, field models require the same t equal to
        history.times[-1]. The model does NOT mutate history. For differentiable
        multi-step generation set create_graph=True and keep the explicit graph.
        Under torch.no_grad(), gradients required by potentials are enabled
        locally; use no_grad, NOT inference_mode, for inference of potential models.
        """
        if torch.is_inference_mode_enabled() and self.needs_history:
            raise RuntimeError("Potential differentiation needs torch.no_grad(), not torch.inference_mode().")
        self._validate_x(queries, history)
        c = _condition(condition, queries.shape[0], self.cfg.condition_dim, queries)
        build_graph = torch.is_grad_enabled() if create_graph is None else create_graph
        with torch.enable_grad():
            if not self.needs_history:
                drift = self._direct_velocity(queries, t, c)
                kind = "node_direct"
            else:
                if not _same_time(t, history.times[-1]):
                    raise ValueError("t must match the last supplied law snapshot. Future history is not allowed.")
                weights = self.temporal.weights(history.times)
                parts = []
                for start in range(0, queries.shape[1], self.cfg.query_chunk):
                    parts.append(self._field_chunk(queries[:, start:start + self.cfg.query_chunk],
                                                   history, c, weights, create_graph=build_graph))
                drift = torch.cat(parts, dim=1)
                kind = "node_pl" if self.spec.temporal != "fir" else "node_fir"
        if not build_graph:
            drift = drift.detach()
        return FieldOutput(drift, self.diffusion, self.spec.model_id, kind)

    def cell_average(self, queries: Tensor, left: Tensor | float, right: Tensor | float,
                     history: Optional[FieldHistory] = None, condition: Optional[Tensor] = None,
                     *, create_graph: Optional[bool] = None) -> FieldOutput:
        """Average field at fixed query x over [left,right], NOT a state update.

        Field models: analytic RL integral of the PC history; last snapshot=left.
        Standard MLP: 3-point Gauss--Legendre time quadrature at fixed x.
        In particular the initial interval uses the real F[rho_0], not zero drift.
        """
        if torch.is_inference_mode_enabled() and self.needs_history:
            raise RuntimeError("Use torch.no_grad(), not inference_mode, with potential fields.")
        self._validate_x(queries, history)
        c = _condition(condition, queries.shape[0], self.cfg.condition_dim, queries)
        l = torch.as_tensor(left, dtype=queries.dtype, device=queries.device)
        r = torch.as_tensor(right, dtype=queries.dtype, device=queries.device)
        if l.ndim or r.ndim or not bool(r > l) or not bool(l >= 0):
            raise ValueError("Require scalar 0<=left<right.")
        build_graph = torch.is_grad_enabled() if create_graph is None else create_graph
        with torch.enable_grad():
            if not self.needs_history:
                mid, half = 0.5 * (l + r), 0.5 * (r - l)
                a = math.sqrt(3.0 / 5.0)
                drift = (5.0 / 18.0 * self._direct_velocity(queries, mid - a * half, c)
                         + 4.0 / 9.0 * self._direct_velocity(queries, mid, c)
                         + 5.0 / 18.0 * self._direct_velocity(queries, mid + a * half, c))
                kind = "cell_average_gauss3"
            else:
                if not _same_time(l, history.times[-1]):
                    raise ValueError("Last accepted source snapshot must be at the left endpoint.")
                weights = self.temporal.weights(history.times, next_time=r)
                parts = []
                for start in range(0, queries.shape[1], self.cfg.query_chunk):
                    parts.append(self._field_chunk(queries[:, start:start + self.cfg.query_chunk],
                                                   history, c, weights, create_graph=build_graph))
                drift = torch.cat(parts, dim=1)
                kind = "cell_average_pc" if self.spec.temporal != "fir" else "cell_average_fir"
        if not build_graph:
            drift = drift.detach()
        return FieldOutput(drift, self.diffusion, self.spec.model_id, kind)

    def differential_fields(self, queries: Tensor, t: Tensor | float,
                            history: Optional[FieldHistory] = None, condition: Optional[Tensor] = None,
                            *, curl_pairs: Optional[Sequence[Tuple[int, int]]] = None,
                            divergence_probes: Optional[Tensor] = None,
                            create_graph: bool = False) -> FieldOutput:
        """Consistent drift/divergence/curl, differentiating w.r.t. state coordinates.

        Holds the empirical law fixed for spatial partial derivatives. D=2 returns
        curl pair (0,1), the ordinary scalar curl. D>2 requires selected pairs or
        computes all pairs only for D<=16. No two-scalar 'curl' in high dimensions.

        divergence_probes: optional externally supplied Rademacher vectors [P,E,Q,D].
            Their contraction estimates trace(Jacobian). No random draws here.
            Without probes, divergence is exact but needs D reverse-mode calls.
        For higher-order training of these diagnostics set create_graph=True.
        """
        d = self.cfg.state_dim
        if curl_pairs is None:
            if d > 16:
                raise ValueError("Choose curl_pairs explicitly for D>16; a full vorticity tensor is expensive.")
            pairs = tuple((i, j) for i in range(d) for j in range(i + 1, d))
        else:
            pairs = tuple((int(i), int(j)) for i, j in curl_pairs)
            if any(not 0 <= i < j < d for i, j in pairs) or len(set(pairs)) != len(pairs):
                raise ValueError("curl_pairs must be unique pairs 0<=i<j<D.")
        with torch.enable_grad():
            x = queries.clone()
            if not x.requires_grad:
                x.requires_grad_(True)
            result = self.forward(x, t, history, condition, create_graph=True)
            v = result.drift
            rows: Dict[int, Tensor] = {}

            def row(i: int) -> Tensor:
                if i not in rows:
                    rows[i] = torch.autograd.grad(v[..., i].sum(), x, create_graph=create_graph,
                                                  retain_graph=True)[0]
                return rows[i]

            if divergence_probes is None:
                divergence = sum((row(i)[..., i] for i in range(d)), torch.zeros_like(v[..., 0]))
                estimator = "exact_spatial_trace"
            else:
                p = divergence_probes
                _real_tensor(p, "divergence_probes")
                if p.ndim != 4 or p.shape[1:] != x.shape or p.shape[0] < 1:
                    raise ValueError("Expected probes[P,E,Q,D].")
                if p.device != x.device or p.dtype != x.dtype:
                    raise ValueError("Probe dtype/device must match queries.")
                traces = []
                for z in p:
                    vjp = torch.autograd.grad(v, x, grad_outputs=z, create_graph=create_graph,
                                              retain_graph=True)[0]
                    traces.append((z * vjp).sum(-1))
                divergence = torch.stack(traces).mean(0)
                estimator = f"external_probe_trace_P={p.shape[0]}"
            curls = [row(j)[..., i] - row(i)[..., j] for i, j in pairs]
            curl = torch.stack(curls, dim=-1) if curls else v.new_empty(*v.shape[:-1], 0)
        if not create_graph:
            v, divergence, curl = v.detach(), divergence.detach(), curl.detach()
        return FieldOutput(v, self.diffusion, self.spec.model_id, result.representation,
                           divergence, curl, pairs, estimator)

    def order_summary(self) -> dict:
        """Logging only; detaching here is intentional. Never use this to compute a field."""
        if not isinstance(self.temporal, RLFieldSpectrum):
            return {"mode": self.spec.temporal}
        if self.spec.temporal == "zero":
            return {"mode": "zero", "orders": [[0.0, 0.0]], "coefficients": [[1.0, 0.0]]}
        a, c = self.temporal.parameters_in_order_space()
        return {"mode": self.spec.temporal,
                "orders": torch.view_as_real(a).detach().cpu().tolist(),
                "coefficients": torch.view_as_real(c).detach().cpu().tolist()}


def build_model(model_id: str, config: ModelConfig, *, complex_structure: Optional[Tensor] = None) -> FlowFieldModel:
    if model_id not in SPECS:
        raise ValueError(f"Unknown model_id={model_id!r}. Use one of {tuple(SPECS)}; old M0--M3 IDs are intentionally not accepted.")
    return FlowFieldModel(SPECS[model_id], config, complex_structure)


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Count real scalar degrees of freedom; no capacity-matching dummy tensors."""
    def count(params: Iterable[Tensor]) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in params)
    return {"trainable_real_scalars": count(p for p in model.parameters() if p.requires_grad),
            "fixed_parameter_real_scalars": count(p for p in model.parameters() if not p.requires_grad),
            "buffer_real_scalars": count(model.buffers())}


def optimization_groups(model: FlowFieldModel, *, network_lr: float = 2e-4,
                        order_lr: float = 2e-5, weight_decay: float = 1e-4) -> list[dict]:
    """Parameter grouping only, NOT optimizer creation or a training loop."""
    order, ordinary = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (order if name.startswith("temporal.") else ordinary).append(p)
    groups = [{"params": ordinary, "lr": network_lr, "weight_decay": weight_decay}]
    if order:
        groups.append({"params": order, "lr": order_lr, "weight_decay": 0.0})
    return groups


@dataclass(frozen=True)
class FlatSequenceLayout:
    """An invertible reshape ONLY. No autoencoder, pooling, padding or PCA."""
    length: int
    channels: int

    @property
    def state_dim(self) -> int:
        return self.length * self.channels

    def flatten(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != (self.length, self.channels):
            raise ValueError("Sequence shape does not match the recorded layout.")
        return x.reshape(*x.shape[:-2], self.state_dim)

    def unflatten(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.state_dim:
            raise ValueError("State dimension does not match the recorded layout.")
        return x.reshape(*x.shape[:-1], self.length, self.channels)


class EndpointTrajectoryMap(nn.Module):
    """Shared deterministic robot trajectory representation, NOT a planner.

    q(s)=q_start+(q_goal-q_start)*(10s^3-15s^4+6s^5)
          + sum_r coeff[r,j] * s^3(1-s)^3 * sin((r+1)*pi*s).

    Position, first derivative and second derivative of the residual vanish at
    endpoints. This property belongs to the COMMON mapping, not one FM model.
    No collision, joint-limit, feasibility or torque guarantee is provided.
    """
    def __init__(self, joints: int, modes: int, length: int = 64) -> None:
        super().__init__()
        if min(joints, modes) < 1 or length < 4:
            raise ValueError("Invalid trajectory dimensions.")
        self.joints, self.modes, self.length = joints, modes, length
        s = torch.linspace(0, 1, length, dtype=torch.float64)
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        r = torch.arange(1, modes + 1, dtype=torch.float64)
        basis = (s**3 * (1-s)**3)[:, None] * torch.sin(math.pi * s[:, None] * r[None, :])
        self.register_buffer("s", s)
        self.register_buffer("blend", blend)
        self.register_buffer("basis", basis)

    @property
    def state_dim(self) -> int:
        return self.joints * self.modes

    def forward(self, coefficients: Tensor, q_start: Tensor, q_goal: Tensor) -> Tensor:
        if coefficients.shape[-1] != self.state_dim:
            raise ValueError("Expected flattened coefficients[...,modes*joints].")
        if q_start.shape != q_goal.shape or q_start.shape != coefficients.shape[:-1] + (self.joints,):
            raise ValueError("Endpoint arrays must match the coefficient batch dimensions.")
        coeff = coefficients.reshape(*coefficients.shape[:-1], self.modes, self.joints)
        basis = self.basis.to(device=coeff.device, dtype=coeff.dtype)
        blend = self.blend.to(device=coeff.device, dtype=coeff.dtype)
        baseline = q_start[..., None, :] + blend[:, None] * (q_goal-q_start)[..., None, :]
        return baseline + torch.einsum("tr,...rj->...tj", basis, coeff)


# No __main__, no runnable training/testing examples, no dataset/network access.
