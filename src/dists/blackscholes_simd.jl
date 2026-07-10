# Batched, SIMD-vectorized fixed-step BS-IV inversion.
#
# Two passes over the batch:
#   1. seed pass (scalar): OTM reduction + regime-dispatched closed-form seed.
#      The regime selection branches per input; it stays scalar and its outputs
#      (seed, κ, cstar, E, invE) are staged into a workspace.
#   2. polish pass (Vec{W,Float64}): exactly N branch-free HH-4 updates on W
#      lanes at once — blended Cody Φ (vecmath.jl), branch-free vexp, and
#      select-based guards. No lane divergence by construction.
#
# Same validity domain as bs_implied_vol_fixed: delta ∈ [0.05, 0.95] — see the
# scalar kernel's docstring. Accuracy differs from the scalar fixed kernel only
# through vexp-vs-Base.exp rounding (~1 ulp).

using SIMD: Vec, vload, vstore

# workspace: staged per-input quantities between the two passes
struct BSFixedWorkspace
    κ::Vector{Float64}
    cstar::Vector{Float64}
    E::Vector{Float64}
    invE::Vector{Float64}
end
BSFixedWorkspace(n::Int) = BSFixedWorkspace(zeros(n), zeros(n), zeros(n), zeros(n))

# one branch-free HH-4 update on a scalar or a Vec lane-bundle
@inline function _hh4_update_bf(v, κ, cstar, E, invE)
    invv = 1.0 / v
    d1 = -κ * invv + 0.5 * v
    d2 = d1 - v
    g1 = vexp(-0.5 * d1 * d1)
    fp = INV_SQRT2PI * g1
    g2 = g1 * invE                      # exp(-d2²/2) via the exact identity
    Φ1 = phi_withg_bf(d1, g1)
    Φ2 = phi_withg_bf(d2, g2)
    f = Φ1 - E * Φ2 - cstar
    r = f / fp
    d1d2 = d1 * d2
    φ2 = d1d2 * invv
    ξ = (d1d2 * d1d2 - (d1 * d1 + d2 * d2) - d1d2) * invv * invv
    denom = -6.0 + r * (6.0 * φ2 - r * ξ)
    vn = v + 3.0 * r * (2.0 - r * φ2) / denom
    vn = sel(abs(denom) < 1e-20, v - r, vn)     # Newton fallback
    vn = sel(isfinite(vn), vn, v)               # non-finite update -> stay
    return min(max(vn, 1e-10), 5.0)             # clamp to solver domain
end

"""
    bs_implied_vol_fixed_batch!(out, ks, cs, ::Val{N}, ::Val{W}; ws)

Batched fixed-step BS-IV: scalar seed pass, then N branch-free HH-4 updates
vectorized W lanes wide (`W ∈ {2,4,8}`; pick the host's native Float64 width).
Preallocate `ws = BSFixedWorkspace(length(ks))` to make repeat calls
allocation-free.
"""
function bs_implied_vol_fixed_batch!(out::Vector{Float64},
                                     ks::Vector{Float64}, cs::Vector{Float64},
                                     ::Val{N}, ::Val{W};
                                     ws::BSFixedWorkspace = BSFixedWorkspace(length(ks))) where {N,W}
    n = length(ks)
    @assert length(cs) == n && length(out) == n && length(ws.κ) >= n
    # --- pass 1: scalar seeds (regime branches live here) ---
    @inbounds for i in 1:n
        κ, cstar, E, invE = _otm_reduce(ks[i], cs[i])
        ws.κ[i] = κ; ws.cstar[i] = cstar; ws.E[i] = E; ws.invE[i] = invE
        out[i] = bs_seed(κ, cstar, E)
    end
    # --- pass 2: branch-free vectorized polish ---
    i = 1
    @inbounds while i + W - 1 <= n
        v     = vload(Vec{W,Float64}, out, i)
        κ     = vload(Vec{W,Float64}, ws.κ, i)
        cstar = vload(Vec{W,Float64}, ws.cstar, i)
        E     = vload(Vec{W,Float64}, ws.E, i)
        invE  = vload(Vec{W,Float64}, ws.invE, i)
        for _ in 1:N
            v = _hh4_update_bf(v, κ, cstar, E, invE)
        end
        vstore(v, out, i)
        i += W
    end
    @inbounds while i <= n                      # remainder lanes, scalar
        v = out[i]
        for _ in 1:N
            v = _hh4_update_bf(v, ws.κ[i], ws.cstar[i], ws.E[i], ws.invE[i])
        end
        out[i] = v
        i += 1
    end
    return out
end
