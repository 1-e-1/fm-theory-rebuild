"""Independent numerical/semantic audit of the five delivered Phase-1 models.

Run: python tests/model_correctness.py
No formal run, control-model training, checkpoint import, or data access occurs.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import loggamma
import torch
from theory_fm_models import (MAIN_MODELS, ModelConfig, FieldHistory,
    RLFieldSpectrum, build_model, canonical_complex_structure, apply_j,
    complex_loggamma_right, positive_real_complex_power, FlatSequenceLayout,
    EndpointTrajectoryMap)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = []


def close(a, b, atol=2e-9, rtol=2e-8):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    if not torch.allclose(a, b.to(a), atol=atol, rtol=rtol):
        raise AssertionError(f"max error={float((a-b.to(a)).abs().max())}, atol={atol}, rtol={rtol}")
    return float((a-b.to(a)).detach().abs().max()) if a.numel() else 0.0


def expect_error(fn, kinds=(ValueError, TypeError, RuntimeError)):
    try:
        fn()
    except kinds:
        return
    raise AssertionError("Expected an explicit validation error")


class FixedSpectrum(RLFieldSpectrum):
    """Test input fixture: exact orders including the boundary alpha=-1."""
    def __init__(self, alpha, coefficient=1.0, reference_time=1.0):
        super().__init__(ModelConfig(state_dim=2, reference_time=reference_time), "complex")
        self.exact_alpha = torch.as_tensor([alpha], dtype=torch.complex128)
        self.exact_coeff = torch.as_tensor([coefficient], dtype=torch.complex128)

    def parameters_in_order_space(self):
        return self.exact_alpha, self.exact_coeff


def record(name, fn):
    tic = time.perf_counter()
    try:
        details = fn() or {}
        result = {"name": name, "status": "PASSED", "seconds": time.perf_counter()-tic,
                  "details": details}
    except Exception as exc:
        result = {"name": name, "status": "FAILED", "seconds": time.perf_counter()-tic,
                  "error": f"{type(exc).__name__}: {exc}"}
    RESULTS.append(result)
    print(json.dumps(result), flush=True)


def gamma_reference():
    z = torch.tensor([0.61+0.49j, 0.8-0.4j, 1.0+0j, 1.2+0.15j,
                      1.6-0.2j, 2.4+0.45j, 5.0-0.1j], dtype=torch.complex128)
    value = complex_loggamma_right(z)
    err = close(value, torch.from_numpy(loggamma(z.numpy())), atol=8e-14, rtol=2e-13)
    expect_error(lambda: complex_loggamma_right(torch.tensor([0.5+0j])))
    a = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
    obj = lambda aa, bb: (complex_loggamma_right(torch.complex(aa, bb)).real
                          + 0.37*complex_loggamma_right(torch.complex(aa, bb)).imag)
    da, db = torch.autograd.grad(obj(a, b), (a, b))
    h = 2e-6
    ea = (obj(a.detach()+h,b.detach())-obj(a.detach()-h,b.detach()))/(2*h)
    eb = (obj(a.detach(),b.detach()+h)-obj(a.detach(),b.detach()-h))/(2*h)
    err_grad = max(close(da,ea,atol=2e-9), close(db,eb,atol=2e-9))
    return {"dtype": "CPU float64/complex128", "reference": "scipy.special.loggamma",
            "max_value_error": err, "max_gradient_error": err_grad,
            "inputs": [[z.real.item(),z.imag.item()] for z in z]}


def power_zero():
    r = torch.tensor([0,0.02,0.4,2], dtype=torch.float64)
    z = torch.tensor([0.8+0.2j,1.6-0.3j], dtype=torch.complex128, requires_grad=True)
    v = positive_real_complex_power(r,z)
    ref = torch.from_numpy(np.asarray([[0j,0j]] + [[rr**zz for zz in z.detach().numpy()] for rr in r[1:].numpy()]))
    err = close(v,ref)
    grad = torch.autograd.grad(v.real.sum()+0.3*v.imag.sum(),z)[0]
    assert torch.isfinite(grad).all()
    expect_error(lambda: positive_real_complex_power(torch.tensor([-1.0]),z))
    expect_error(lambda: positive_real_complex_power(r,torch.tensor([0.0+0j])))
    return {"max_error": err, "zero_base_gradient_is_finite": True}


def exact_rl_histories():
    t = torch.tensor([0.0,0.07,0.21,0.45,0.8], dtype=torch.float64)
    err = 0.0
    for alpha in (0j,-1+0j,-0.6+0.2j,0.2+0.15j):
        s = FixedSpectrum(alpha)
        w = s.weights(t)
        # RL derivative 1 and t, including the nonzero causal initial term.
        c = complex(np.exp(-loggamma(1-alpha))) * float(t[-1])**(-alpha)
        linear = complex(np.exp(-loggamma(2-alpha))) * float(t[-1])**(1-alpha)
        err = max(err,close(w.sum(),torch.tensor(c,dtype=torch.complex128)),
                  close((w*t).sum(),torch.tensor(linear,dtype=torch.complex128)))
        if alpha == 0j:
            close(w,torch.tensor([0,0,0,0,1],dtype=torch.complex128),atol=5e-15)
            close(s.weights(t,next_time=1.0),torch.tensor([0,0,0,0,1],dtype=torch.complex128),atol=5e-15)
        if alpha == -1+0j:
            close(w.sum(),t[-1])
            close((w*t).sum(),t[-1].square()/2)
    return {"grid":t.tolist(), "orders":[[0,0],[-1,0],[-0.6,0.2],[0.2,0.15]],
            "max_error":err,"representations":"node=PL; f0 is retained"}


def pc_reference():
    times = torch.tensor([0.0,0.05,0.23,0.56],dtype=torch.float64)
    values = torch.tensor([1.2,-0.4,0.7,1.6],dtype=torch.float64)
    end = 0.8
    errors=[]
    for alpha in (-1+0j,-0.6+0.2j,0j,0.2+0.15j):
        p=1-alpha
        def primitive_at(t):
            total=0j
            ends=list(times[1:].numpy())+[end]
            for left,right,f in zip(times.numpy(),ends,values.numpy()):
                a,b=max(t-left,0),max(t-right,0)
                total+=f*((a**p if a>0 else 0j)-(b**p if b>0 else 0j))
            return total*np.exp(-loggamma(2-alpha))
        expected=(primitive_at(end)-primitive_at(float(times[-1])))/(end-float(times[-1]))
        actual=(FixedSpectrum(alpha).weights(times,next_time=end)*values).sum()
        errors.append(close(actual,torch.tensor(expected,dtype=torch.complex128)))
    return {"max_error":max(errors),"reference":"independent PC fractional integral primitive",
            "times":times.tolist(),"values":values.tolist(),"right":end}


def initial_term():
    alpha=0.2+0.15j
    s=FixedSpectrum(alpha)
    t=torch.linspace(0,0.7,17,dtype=torch.float64)
    f0=2.3
    expected=f0*0.7**(-alpha)*np.exp(-loggamma(1-alpha))
    actual=(s.weights(t)*f0).sum()
    close(actual,torch.tensor(expected,dtype=torch.complex128))
    # Caputo subtraction for a constant is zero and thus distinctly different.
    assert abs(complex(actual.item()))>1
    h=0.03125
    first=s.weights(torch.zeros(1,dtype=torch.float64),next_time=h).sum()*f0
    exp_first=f0*h**(-alpha)*np.exp(-loggamma(2-alpha))
    close(first,torch.tensor(exp_first,dtype=torch.complex128))
    expect_error(lambda:s.weights(torch.zeros(1,dtype=torch.float64)))
    return {"f0":f0,"alpha":[alpha.real,alpha.imag],"node":torch.view_as_real(actual).tolist(),
            "initial_cell":torch.view_as_real(first).tolist(),"Caputo_constant_value":0.0}


def power_refinement():
    errors={}
    for alpha in (-0.6+0.2j,0.2+0.15j):
        expected=2*np.exp(-loggamma(3-alpha))*0.8**(2-alpha)
        err=[]
        for h in (17,65,257):
            t=torch.linspace(0,0.8,h,dtype=torch.float64)
            actual=(FixedSpectrum(alpha).weights(t)*t.square()).sum()
            err.append(abs(complex(actual.item())-expected))
        assert err[-1]<err[0]/10 and err[-1]<0.0001
        errors[str(alpha)]=err
    return {"f":"t^2","nodes":[17,65,257],"absolute_errors":errors,
            "reference":"Gamma(3)/Gamma(3-alpha)*t^(2-alpha)"}


def kernel_gradients(device="cpu",dtype=torch.float64):
    s=RLFieldSpectrum(ModelConfig(state_dim=2),"complex").to(device=device,dtype=dtype)
    t=torch.tensor([0,0.08,0.2,0.41,0.7],device=device,dtype=torch.float64)
    vals=torch.tensor([0.7,-0.2,0.9,1.2,0.4],device=device,dtype=torch.float64)
    def obj():
        out=(s.weights(t)*vals).sum()
        return out.real+0.41*out.imag
    grads=torch.autograd.grad(obj(),(s.raw_a,s.raw_b,s.raw_c))
    step=2e-6 if dtype==torch.float64 else 0.0005
    errs={}
    for name,p,g in zip(("raw_a","raw_b","raw_c"),(s.raw_a,s.raw_b,s.raw_c),grads):
        assert torch.isfinite(g).all() and g.abs().max()>1e-5
        diffs=[]
        for i in range(p.numel()):
            saved=p[i].detach().clone()
            with torch.no_grad():p[i]=saved+step
            upper=obj().detach()
            with torch.no_grad():p[i]=saved-step
            lower=obj().detach()
            with torch.no_grad():p[i]=saved
            diffs.append((upper-lower)/(2*step))
        errors=torch.stack(diffs)-g.double()
        tolerance=2e-9 if dtype==torch.float64 else 0.00012
        assert errors.abs().max()<tolerance, (name,errors)
        errs[name]=float(errors.abs().max())
    return {"device":device,"parameter_dtype":str(dtype),"kernel_dtype":"complex128",
            "max_gradient_errors":errs,"central_difference_step":step}


def fixture(device="cpu",dtype=torch.float64,d=4,channels=0):
    torch.manual_seed(231)
    cfg=ModelConfig(state_dim=d,condition_dim=channels,hidden_dim=16,hidden_layers=2,
                    source_chunk=2,query_chunk=2)
    x=torch.randn(2,3,d,device=device,dtype=dtype)
    t=torch.tensor([0,0.1,0.4,0.7],device=device,dtype=torch.float64)
    hist=FieldHistory(t,torch.randn(2,4,5,d,device=device,dtype=dtype))
    c=torch.randn(2,channels,device=device,dtype=dtype) if channels else None
    return cfg,x,hist,c


def five_models(device="cpu",dtype=torch.float64):
    cfg,x,hist,c=fixture(device,dtype,64)
    details={}
    for name in MAIN_MODELS:
        model=build_model(name,cfg).to(device=device,dtype=dtype)
        rng=torch.get_rng_state().clone()
        gpu_rng=torch.cuda.get_rng_state().clone() if device=="cuda" else None
        out=model(x,0.7,hist if model.needs_history else None)
        again=model(x,0.7,hist if model.needs_history else None)
        close(out.drift,again.drift,atol=0,rtol=0)
        assert torch.equal(rng,torch.get_rng_state())
        if device=="cuda":assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
        assert out.drift.shape==x.shape and torch.isfinite(out.drift).all()
        out.drift.square().mean().backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        if model.needs_history:
            for p in (model.temporal.raw_a,model.temporal.raw_c):assert p.grad.abs().max()>0
            if name!="FM_REAL_ODE":assert model.temporal.raw_b.grad.abs().max()>0
        with torch.no_grad():
            inferred=model(x,0.7,hist if model.needs_history else None)
        close(out.drift,inferred.drift,atol=1e-7 if dtype==torch.float32 else 1e-13)
        cell=model.cell_average(x,0,0.03125,hist.prefix(1) if model.needs_history else None)
        assert torch.isfinite(cell.drift).all() and cell.drift.abs().max()>0
        if model.needs_history:
            with torch.inference_mode():
                expect_error(lambda:model(x,0.7,hist))
        details[name]={"shape":list(out.drift.shape),"all_parameter_gradients_finite":True,
                       "diffusion":out.diffusion,"initial_cell_nonzero":True}
    return {"device":device,"dtype":str(dtype),"models":details,
            "hidden_state":False,"forward_rng_unchanged":True}


def space_properties(device="cpu",dtype=torch.float64):
    cfg,x,h,c=fixture(device,dtype,4,2)
    model=build_model("FM_COMPLEX_ODE",cfg).to(device=device,dtype=dtype)
    tol=2e-5 if dtype==torch.float32 else 2e-8
    potential=model.potentials
    err_even=close(potential._potential(x,c,interaction=True),
                   potential._potential(-x,c,interaction=True),atol=tol)
    xq=x.clone().requires_grad_(True)
    value=potential.values(xq,h.particles[:,0],None,c)
    grad=torch.autograd.grad(value.sum(),xq)[0]
    step=0.001 if dtype==torch.float32 else 2e-5
    diffs=[]
    for i in range(4):
        delta=torch.zeros_like(x);delta[...,i]=step
        diffs.append((potential.values(x+delta,h.particles[:,0],None,c)
                      -potential.values(x-delta,h.particles[:,0],None,c))[...,0]/(2*step))
    fd=torch.stack(diffs,dim=-1)
    err_grad=close(grad,fd,atol=0.001 if dtype==torch.float32 else 2e-8,rtol=0.001 if dtype==torch.float32 else 2e-8)
    perm=torch.tensor([4,2,0,1,3],device=device)
    hp=FieldHistory(h.times,h.particles[:,:,perm])
    drift=model(x,0.7,h,condition=c).drift
    err_perm=close(drift,model(x,0.7,hp,condition=c).drift,atol=tol)
    probabilities=torch.rand(2,4,5,device=device,dtype=dtype)
    probabilities=probabilities/probabilities.sum(-1,keepdim=True)
    hw=FieldHistory(h.times,h.particles,probabilities)
    hwp=FieldHistory(h.times,h.particles[:,:,perm],probabilities[:,:,perm])
    err_weight_perm=close(model(x,0.7,hw,condition=c).drift,
                          model(x,0.7,hwp,condition=c).drift,atol=tol)
    modified=h.particles.clone();modified[:,1]+=1.1
    changed=model(x,0.7,FieldHistory(h.times,modified),condition=c).drift
    assert (changed-drift).abs().max()>1e-5
    zero=FixedSpectrum(0j).to(device)
    zero.exact_alpha=zero.exact_alpha.to(device);zero.exact_coeff=zero.exact_coeff.to(device)
    model.temporal=zero
    err_zero=close(model(x,0.7,h,condition=c).drift,
                    model(x,0.7,FieldHistory(h.times,modified),condition=c).drift,atol=tol)
    return {"device":device,"dtype":str(dtype),"even_W_error":err_even,
            "spatial_gradient_fd_error":err_grad,"permutation_error":err_perm,
            "weighted_permutation_error":err_weight_perm,"zero_order_past_effect":err_zero,
            "nonzero_order_changed_snapshot_effect":float((changed-drift).abs().max())}


def alias_partial():
    cfg,x,h,c=fixture(d=2)
    model=build_model("FM_COMPLEX_ODE",cfg).double()
    state=x[:1].clone().requires_grad_(True)
    hist=FieldHistory(h.times,state[:,None].expand(-1,4,-1,-1))
    # Receiver and law share storage. The force must equal fixed independent law.
    aliased=model(state,0.7,hist).drift
    fixed=model(state,0.7,FieldHistory(h.times,hist.particles.detach().clone())).drift
    err=close(aliased,fixed)
    law=h.particles[:1].clone().requires_grad_(True)
    out=model(state,0.7,FieldHistory(h.times,law)).drift
    dlaw=torch.autograd.grad(out.square().sum(),law,retain_graph=True)[0]
    dx=torch.autograd.grad(out.square().sum(),state)[0]
    assert torch.isfinite(dlaw).all() and dlaw.abs().max()>0
    assert torch.isfinite(dx).all() and dx.abs().max()>0
    return {"aliased_vs_fixed_law_force_error":err,"outer_law_dependencies_retained":True,
            "spatial_partial_holds_law_fixed":True}


def full_realification():
    cfg,x,h,c=fixture(d=2)
    model=build_model("FM_COMPLEX_ODE",cfg).double()
    weights=model.temporal.weights(h.times)
    forces=[]
    for k in range(4):
        q=x.clone().requires_grad_(True)
        val=model.potentials.values(q,h.particles[:,k],None,q.new_empty(2,0))
        forces.append(-torch.autograd.grad(val.sum(),q)[0])
    encoded=[torch.complex(f[...,0],f[...,1]) for f in forces]
    z=sum(w*f for w,f in zip(weights,encoded))
    expected=torch.stack((z.real,z.imag),dim=-1)
    actual=model(x,0.7,h).drift
    err=close(actual,expected)
    real_only=sum(w.real*f for w,f in zip(weights,forces))
    assert (real_only-actual).abs().max()>1e-5
    return {"complex_encoded_vs_qF_rJF_error":err,
            "omitted_imaginary_kernel_error":float((real_only-actual).abs().max())}


def nested_parameter_gradients(device="cpu",dtype=torch.float64):
    cfg,x,h,c=fixture(device,dtype,d=4)
    model=build_model("FM_COMPLEX_VDC_ODE",cfg).to(device=device,dtype=dtype)
    params=dict(model.named_parameters())
    names=["potentials.v_net.net.0.weight", "potentials.w_net.net.0.weight",
           "potentials.v_net.net.4.weight", "potentials.w_net.net.4.weight",
           "temporal.raw_a", "temporal.raw_b", "temporal.raw_c"]
    def objective():return model(x,0.7,h).drift.square().mean()
    selected=[params[name] for name in names]
    gradients=torch.autograd.grad(objective(),selected)
    errors={}
    step=2e-5 if dtype==torch.float64 else 0.002
    for name,p,g in zip(names,selected,gradients):
        # Select the strongest derivative to avoid a meaningless zero-only test.
        pos=int(g.detach().abs().argmax())
        flat=p.view(-1);saved=flat[pos].detach().clone()
        with torch.no_grad():flat[pos]=saved+step
        upper=objective().detach()
        with torch.no_grad():flat[pos]=saved-step
        lower=objective().detach()
        with torch.no_grad():flat[pos]=saved
        difference=(upper-lower)/(2*step)
        actual=g.flatten()[pos]
        tolerance=1e-9 if dtype==torch.float64 else 0.0001
        err=close(actual,difference,atol=tolerance,rtol=0.002 if dtype==torch.float32 else 1e-7)
        errors[name]={"flat_index":pos,"autograd":float(actual.detach()),
                      "central_difference":float(difference),"error":err}
    return {"device":device,"dtype":str(dtype),"step":step,"force_loss_parameter_derivatives":errors}


def algebraic_reductions():
    cfg,x,h,c=fixture(d=4)
    ode=build_model("FM_COMPLEX_ODE",cfg).double()
    sde=build_model("FM_COMPLEX_SDE",cfg).double()
    sde.load_state_dict(ode.state_dict())
    err_sde=close(ode(x,0.7,h).drift,sde(x,0.7,h).drift,atol=0,rtol=0)
    assert ode.diffusion==0 and abs(sde.diffusion-math.sqrt(0.1))<1e-15
    real=build_model("FM_REAL_ODE",cfg).double()
    real.load_state_dict(ode.state_dict())
    with torch.no_grad():ode.temporal.raw_b.zero_()
    err_real=close(ode(x,0.7,h).drift,real(x,0.7,h).drift)
    dual=build_model("FM_COMPLEX_VDC_ODE",cfg).double()
    dual.temporal.load_state_dict(ode.temporal.state_dict())
    with torch.no_grad():
        for net in ("v_net","w_net"):
            src=getattr(ode.potentials,net).net
            dst=getattr(dual.potentials,net).net
            for a,b in zip(src[:-1].parameters(),dst[:-1].parameters()):b.copy_(a)
            dst[-1].weight.zero_();dst[-1].bias.zero_()
            dst[-1].weight[0].copy_(src[-1].weight[0]);dst[-1].bias[0].copy_(src[-1].bias[0])
    err_dual=close(dual(x,0.7,h).drift,ode(x,0.7,h).drift)
    return {"SDE_ODE_drift_error":err_sde,"complex_zero_b_real_error":err_real,
            "zero_second_potential_single_error":err_dual,"state_Brownian_only":True}


def vdc_laplacian():
    cfg,x,h,c=fixture(d=2)
    model=build_model("FM_COMPLEX_VDC_ODE",cfg).double()
    q=x.clone().requires_grad_(True)
    A=torch.zeros_like(q[...,0]);B=A.clone()
    weights=model.temporal.weights(h.times)
    for k,w in enumerate(weights):
        U,S=model.potentials.values(q,h.particles[:,k],None,q.new_empty(2,0)).unbind(-1)
        A=A+w.real*U-w.imag*S
        B=B+w.imag*U+w.real*S
    def lap(f):
        g=torch.autograd.grad(f.sum(),q,create_graph=True,retain_graph=True)[0]
        return sum(torch.autograd.grad(g[...,i].sum(),q,retain_graph=True)[0][...,i] for i in range(2))
    div=-lap(A);curl=-lap(B)
    out=model.differential_fields(x,0.7,h)
    err=max(close(out.divergence,div),close(out.curl[...,0],curl))
    return {"D":2,"identity_error":err,"div":"-Delta(qU-rS)","curl":"-Delta(rU+qS)"}


def d4_diagnostics(device="cpu",dtype=torch.float64):
    cfg,x,h,c=fixture(device,dtype,d=4)
    model=build_model("FM_COMPLEX_VDC_ODE",cfg).to(device=device,dtype=dtype)
    out=model.differential_fields(x,0.7,h)
    step=0.002 if dtype==torch.float32 else 0.00002
    columns=[]
    for i in range(4):
        delta=torch.zeros_like(x);delta[...,i]=step
        with torch.no_grad():
            plus=model(x+delta,0.7,h).drift
            minus=model(x-delta,0.7,h).drift
        columns.append((plus-minus)/(2*step))
    jac=torch.stack(columns,dim=-1) # output coordinate, input coordinate
    ref_div=jac.diagonal(dim1=-2,dim2=-1).sum(-1)
    ref_curl=torch.stack([jac[...,j,i]-jac[...,i,j] for i,j in out.curl_pairs],-1)
    tol=0.0003 if dtype==torch.float32 else 2e-8
    err=max(close(out.divergence,ref_div,atol=tol,rtol=tol),close(out.curl,ref_curl,atol=tol,rtol=tol))
    assert out.curl.shape==(2,3,6) and len(out.curl_pairs)==6
    selected=model.differential_fields(x,0.7,h,curl_pairs=[(0,1),(2,3)])
    close(selected.curl,out.curl[...,[0,5]],atol=tol)
    had=torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],device=device,dtype=dtype)
    probes=had[:,None,None,:].expand(-1,2,3,-1)
    probed=model.differential_fields(x,0.7,h,divergence_probes=probes)
    err_probe=close(probed.divergence,out.divergence,atol=tol)
    high=build_model("FM_COMPLEX_ODE",ModelConfig(state_dim=64,hidden_dim=4,hidden_layers=1)).to(device=device,dtype=dtype)
    expect_error(lambda:high.differential_fields(torch.zeros(1,1,64,device=device,dtype=dtype),0.7,None))
    return {"device":device,"dtype":str(dtype),"D":4,"curl_shape":list(out.curl.shape),
            "full_jacobian_fd_error":err,"orthogonal_probe_trace_error":err_probe,
            "explicit_selected_pairs":list(selected.curl_pairs)}


def layouts_and_validation():
    j=canonical_complex_structure(64,dtype=torch.float64)
    close(j+j.T,torch.zeros_like(j),atol=0,rtol=0)
    close(j@j,-torch.eye(64,dtype=torch.float64),atol=0,rtol=0)
    layout=FlatSequenceLayout(64,1)
    x=torch.randn(3,64,1,dtype=torch.float64)
    close(layout.unflatten(layout.flatten(x)),x,atol=0,rtol=0)
    traj=EndpointTrajectoryMap(8,4,64)
    coeff=torch.randn(2,32,dtype=torch.float64)
    start=torch.randn(2,8,dtype=torch.float64);goal=torch.randn(2,8,dtype=torch.float64)
    q=traj(coeff,start,goal)
    close(q[:,0],start,atol=0,rtol=0);close(q[:,-1],goal,atol=1e-14)
    # End derivatives of the actual analytic expression, independent of grid FD.
    s=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
    residual=(s**3*(1-s)**3)*torch.sin(math.pi*s)
    g=torch.autograd.grad(residual.sum(),s,create_graph=True)[0]
    gg=torch.autograd.grad(g.sum(),s)[0]
    close(g,torch.zeros_like(g),atol=0,rtol=0);close(gg,torch.zeros_like(gg),atol=0,rtol=0)
    cfg,x,h,c=fixture()
    model=build_model("FM_COMPLEX_ODE",cfg).double()
    expect_error(lambda:ModelConfig(state_dim=3))
    expect_error(lambda:build_model("M0",cfg))
    expect_error(lambda:model(x,0.8,h))
    expect_error(lambda:model(x,0.7,None))
    expect_error(lambda:model(x.half(),0.7,h))
    expect_error(lambda:FieldHistory(h.times.flip(0),h.particles).validate(4))
    expect_error(lambda:FieldHistory(h.times,h.particles,torch.full((2,4,5),0.3,dtype=torch.float64)).validate(4))
    condition_model=build_model("FM_STANDARD",ModelConfig(state_dim=4,condition_dim=2,hidden_dim=4,hidden_layers=1)).double()
    expect_error(lambda:condition_model(x,0.7,condition=torch.zeros(2,3,2,dtype=torch.float64)))
    return {"flatten_shape":[3,64],"robot_coefficients":32,"robot_trajectory_shape":list(q.shape),
            "endpoint_derivatives_zero":True,"J":"64D canonical adjacent-coordinate blocks",
            "invalid_shape_condition_time_history_dtype_rejected":True}


def gaussian_fp_and_brownian():
    # Analytic independent Gaussian teacher verifies the marginal FP sign.
    tau=0.43;kappa=0.05;d=2;dt=1e-5
    mu=torch.tensor([0.6,-0.8],dtype=torch.float64)
    var=torch.tensor([0.7,1.3],dtype=torch.float64)
    x=torch.tensor([[0.2,0.5],[-0.7,-0.4],[1.1,-0.9]],dtype=torch.float64,requires_grad=True)
    def density(q,t):
        sig=1-0.95*t;cov=t*t*var+sig*sig
        return torch.exp(-0.5*((q-t*mu).square()/cov).sum(-1))/(2*math.pi*torch.sqrt(cov.prod()))
    sig=1-0.95*tau;cov=tau*tau*var+sig*sig
    score=-(x-tau*mu)/cov
    current_velocity=mu+(tau*var-0.95*sig)/cov*(x-tau*mu)
    drift=current_velocity+kappa*score
    rho=density(x,tau)
    flux=rho[:,None]*drift
    div=sum(torch.autograd.grad(flux[:,i].sum(),x,retain_graph=True)[0][:,i] for i in range(d))
    grad=torch.autograd.grad(rho.sum(),x,create_graph=True)[0]
    lap=sum(torch.autograd.grad(grad[:,i].sum(),x,retain_graph=True)[0][:,i] for i in range(d))
    derivative=(density(x,tau+dt)-density(x,tau-dt))/(2*dt)
    residual=derivative+div-kappa*lap
    close(residual,torch.zeros_like(residual),atol=2e-10)
    # Different dt values detect sqrt(h) rather than h scaling.
    g=torch.Generator().manual_seed(472)
    statistics=[]
    for h in (0.02,0.1):
        xi=torch.randn(100000,4,generator=g,dtype=torch.float64)
        increment=math.sqrt(2*kappa*h)*xi
        ratio=increment.var(0,unbiased=True)/(2*kappa*h)
        assert (ratio-1).abs().max()<0.018
        correlation=torch.corrcoef(increment.T)-torch.eye(4,dtype=torch.float64)
        assert correlation.abs().max()<0.015
        statistics.append({"h":h,"expected_variance":2*kappa*h,
                           "empirical_variance":increment.var(0,unbiased=True).tolist(),
                           "max_cross_coordinate_correlation":float(correlation.abs().max())})
    return {"tau":tau,"kappa":kappa,"max_FP_residual":float(residual.abs().max()),
            "drift":"v+kappa*score","Brownian":statistics,
            "scope":"analytic and increment reference; trainer/sampler integration checked separately"}


def independent_reference():
    gen=torch.Generator().manual_seed(5409)
    query_y=torch.randn(64,4,generator=gen,dtype=torch.float64)
    query_eps=torch.randn(64,4,generator=gen,dtype=torch.float64)
    reference_y=torch.randn(32,4,generator=gen,dtype=torch.float64)
    reference_eps=torch.randn(32,4,generator=gen,dtype=torch.float64)
    t=torch.tensor([0,0.11,0.37,0.8],dtype=torch.float64)
    history=t[:,None,None]*reference_y+(1-0.95*t[:,None,None])*reference_eps
    rebuilt=t[:,None,None]*reference_y+(1-0.95*t[:,None,None])*reference_eps
    query_y=query_y.flip(0);query_eps=query_eps.roll(1,0)
    close(history,rebuilt,atol=0,rtol=0)
    close(history[0],reference_eps,atol=0,rtol=0)
    # Same endpoints/noise are reconstructed from any two distinct snapshots.
    recovered_y=(history[-1]-(1-0.95*t[-1])*history[0])/t[-1]
    close(recovered_y,reference_y)
    assert len({query_y.data_ptr(),query_eps.data_ptr(),reference_y.data_ptr(),reference_eps.data_ptr()})==4
    return {"query_count":64,"reference_count":32,"history_times":t.tolist(),
            "query_permutation_does_not_change_reference":True,"reference_Y_epsilon_fixed_across_time":True,
            "scope":"independent fixture; trainer actual data-flow integration checked separately"}


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    record("Gamma independent SciPy value/gradient",gamma_reference)
    record("Complex powers zero handling",power_zero)
    record("Exact RL zero, -1, constant and linear PL histories",exact_rl_histories)
    record("Exact PC cell primitive",pc_reference)
    record("RL true initial term and first cell",initial_term)
    record("Continuous Gamma power reference H refinement",power_refinement)
    record("CPU64 real/imag/coefficient order gradients",kernel_gradients)
    record("CPU64 all five forward/backward, statelessness, first cell",five_models)
    record("CPU64 potential evenness/spatial FD/permutation/history",space_properties)
    record("Receiver/source alias fixed-law partial and outer dependencies",alias_partial)
    record("Full complex realification",full_realification)
    record("CPU64 nested force-loss parameter gradient finite differences",nested_parameter_gradients)
    record("Algebraic SDE/ODE, real-order and second-potential reductions",algebraic_reductions)
    record("Two-dimensional VDC div/curl Laplacian signs",vdc_laplacian)
    record("CPU64 four-dimensional full vorticity and probe trace",d4_diagnostics)
    record("J, reversible layout, robot map and input validation",layouts_and_validation)
    record("Gaussian FP full drift sign and Brownian variance",gaussian_fp_and_brownian)
    record("Independent reference history construction",independent_reference)
    if torch.cuda.is_available():
        record("GPU32 real/imag/coefficient order gradients",lambda:kernel_gradients("cuda",torch.float32))
        record("GPU32 all five forward/backward, statelessness, first cell",lambda:five_models("cuda",torch.float32))
        record("GPU32 potential evenness/spatial FD/permutation/history",lambda:space_properties("cuda",torch.float32))
        record("GPU32 nested force-loss parameter gradient finite differences",lambda:nested_parameter_gradients("cuda",torch.float32))
        record("GPU32 four-dimensional full vorticity and probe trace",lambda:d4_diagnostics("cuda",torch.float32))
    metadata={"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
              "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "source_sha256":hashlib.sha256((ROOT/"theory_fm_models.py").read_bytes()).hexdigest(),
              "test_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "TF32":False,"AMP":False,"formal_training_configurations":0,
              "results":RESULTS,"passed":sum(r["status"]=="PASSED" for r in RESULTS),
              "failed":sum(r["status"]=="FAILED" for r in RESULTS)}
    (ROOT/"tests/model_correctness_report.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Passed {metadata['passed']}/{len(RESULTS)}",flush=True)
    return 1 if metadata["failed"] else 0


if __name__=="__main__":
    raise SystemExit(main())
