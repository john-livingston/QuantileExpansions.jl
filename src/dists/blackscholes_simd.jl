# Batched, SIMD-vectorized fixed-step BS-IV inversion — fully fused.
#
# Everything is branch-free and generic over Float64 / Vec{W,Float64}: the OTM
# reduction (vexp), the regime seed (all five regime candidates evaluated as
# rationals + sqrt on every lane — including the deep/Mills seed via the
# blended-Acklam norminv_bf — then lane-selected), and exactly N HH-4 updates
# (blended Cody Φ). One pass: load W lanes → reduce → seed → polish → store.
# No lane divergence and no intermediate workspace.
#
# Same validity domain as bs_implied_vol_fixed: delta ∈ [0.05, 0.95] — see the
# scalar kernel's docstring. Differs from the scalar fixed kernel only through
# vexp/vlog-vs-libm rounding, which the quartic polish contracts below 1e-13.

using SIMD: Vec, vload, vstore

# branch-free OTM reduction (mirrors _otm_reduce)
@inline function _otm_reduce_bf(k, c)
    κ = abs(k)
    ek = vexp(k)
    invek = 1.0 / ek
    m = k >= 0.0
    cstar = max(sel(m, c, invek * (c - 1.0 + ek)), 1e-300)
    E = sel(m, ek, invek)
    invE = sel(m, invek, ek)
    return κ, cstar, E, invE
end

# branch-free regime seed (mirrors bs_seed lane-for-lane; every candidate is
# computed on every lane, non-finite candidates fall back to the ATM seed
# before selection — the scalar path never produces them on valid inputs)
@inline function bs_seed_bf(κ, cstar, E)
    # ATM Taylor seed
    s = SQRT2PI * cstar
    s2 = s * s
    vatm = s * (1.0 + s2 * (1.0/24.0 + s2 * (7.0/1920.0 + s2 * (127.0/322560.0))))
    # P1 quadratic seed
    φ = κ * (1.0 + κ * (0.5 + κ * (1.0/6.0 + κ * (1.0/24.0))))
    twocφ = 2.0 * cstar + φ
    Nd = twocφ * twocφ - 8.0 * A1 * A1 * κ * φ * (2.0 + φ)
    sq = sqrt(max(Nd, 0.0))                       # discriminant clip, branch-free
    v1 = (twocφ + sq) / (2.0 * A1 * (2.0 + φ))
    # one shared Newton step on the P3 and P7 surrogates at v1
    invv = 1.0 / max(v1, 1e-10)
    d1 = -κ * invv + 0.5 * v1
    d2 = d1 - v1
    base = 0.5 * (1.0 - E)
    d1p = -d2 * invv
    d2p = -d1 * invv
    cP7  = base + A1 * (_poly_S(d1, Val(7)) - E * _poly_S(d2, Val(7)))
    dcP7 = A1 * (_poly_dS(d1, Val(7)) * d1p - E * _poly_dS(d2, Val(7)) * d2p)
    v7 = v1 - (cP7 - cstar) / dcP7
    cP3  = base + A1 * (_poly_S(d1, Val(3)) - E * _poly_S(d2, Val(3)))
    dcP3 = A1 * (_poly_dS(d1, Val(3)) * d1p - E * _poly_dS(d2, Val(3)) * d2p)
    v3 = v1 - (cP3 - cstar) / dcP3
    # deep-OTM Mills seed (norminv_bf is branch-free)
    z = norminv_bf(cstar)
    vq = z + sqrt(z * z + 2.0 * κ)
    ρ = vq * vq / max(κ + 0.5 * vq * vq, 1e-300)
    arg = min(cstar / max(ρ, 1e-300), 0.999999)
    zq = norminv_bf(arg)
    vdeep = zq + sqrt(zq * zq + 2.0 * κ)
    # sanitize candidates, then select the regime (mirrors bs_seed's branches);
    # |d2(v1)| past D2_VALID leaves the polynomial surrogate's basin -> Mills seed
    # (the lane-friendly replacement for the (κ, c*)-proxy tail filter).
    vpoly = sel(κ <= Κ2, v7, sel(κ <= Κ3, 0.5 * (v3 + v7), v3))
    vpoly = sel(isfinite(vpoly), vpoly, vatm)
    vdeep = sel(isfinite(vdeep), vdeep, vatm)
    use_poly = (abs(d2) <= D2_VALID) & (κ <= Κ4)
    vsel = sel(use_poly, max(vpoly, vatm), max(vdeep, vatm))
    return sel(κ < Κ1, vatm, vsel)
end

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

# reduce → seed → N updates, on one scalar or one Vec bundle
@inline function _solve_lanes(k, c, ::Val{N}) where {N}
    κ, cstar, E, invE = _otm_reduce_bf(k, c)
    v = bs_seed_bf(κ, cstar, E)
    for _ in 1:N
        v = _hh4_update_bf(v, κ, cstar, E, invE)
    end
    return v
end

# workspace for the two-pass (scalar-seed) strategy
struct BSFixedWorkspace
    κ::Vector{Float64}
    cstar::Vector{Float64}
    E::Vector{Float64}
    invE::Vector{Float64}
end
BSFixedWorkspace(n::Int) = BSFixedWorkspace(zeros(n), zeros(n), zeros(n), zeros(n))

# strategy 1 — fused: reduce+seed+polish all vectorized, single pass.
# Wins when lanes are wide enough to cover the blended seed's ~3x work
# (all five regime candidates + two norminv_bf on every lane).
function _batch_fused!(out, ks, cs, ::Val{N}, ::Val{W}) where {N,W}
    n = length(ks)
    i = 1
    @inbounds while i + W - 1 <= n
        k = vload(Vec{W,Float64}, ks, i)
        c = vload(Vec{W,Float64}, cs, i)
        vstore(_solve_lanes(k, c, Val(N)), out, i)
        i += W
    end
    @inbounds while i <= n
        out[i] = _solve_lanes(ks[i], cs[i], Val(N))
        i += 1
    end
    return out
end

# strategy 2 — two-pass: scalar branchy seed staged through a workspace, then
# the vectorized polish. Wins on narrow SIMD (e.g. 2-wide NEON), where the
# scalar regime dispatch is cheaper than the blended vector seed.
function _batch_twopass!(out, ks, cs, ::Val{N}, ::Val{W}, ws::BSFixedWorkspace) where {N,W}
    n = length(ks)
    @assert length(ws.κ) >= n
    @inbounds for i in 1:n
        κ, cstar, E, invE = _otm_reduce(ks[i], cs[i])
        ws.κ[i] = κ; ws.cstar[i] = cstar; ws.E[i] = E; ws.invE[i] = invE
        out[i] = bs_seed(κ, cstar, E)
    end
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
    @inbounds while i <= n
        v = out[i]
        for _ in 1:N
            v = _hh4_update_bf(v, ws.κ[i], ws.cstar[i], ws.E[i], ws.invE[i])
        end
        out[i] = v
        i += 1
    end
    return out
end

"""
    bs_implied_vol_fixed_batch!(out, ks, cs, ::Val{N}, ::Val{W};
                                vector_seed = (Sys.ARCH === :x86_64), ws = nothing)

Batched fixed-step BS-IV, vectorized W lanes wide. Two seed strategies:
`vector_seed = true` runs the fully-fused branch-free path (best on wide SIMD,
e.g. AVX-512); `false` uses a scalar seed pass staged through `ws` (best on
2-wide NEON — pass a preallocated `BSFixedWorkspace(length(ks))` to make repeat
calls allocation-free; the fused path never allocates).
"""
function bs_implied_vol_fixed_batch!(out::Vector{Float64},
                                     ks::Vector{Float64}, cs::Vector{Float64},
                                     ::Val{N}, ::Val{W};
                                     vector_seed::Bool = (Sys.ARCH === :x86_64),
                                     ws::Union{Nothing,BSFixedWorkspace} = nothing) where {N,W}
    n = length(ks)
    @assert length(cs) == n && length(out) == n
    if vector_seed
        return _batch_fused!(out, ks, cs, Val(N), Val(W))
    else
        return _batch_twopass!(out, ks, cs, Val(N), Val(W),
                               ws === nothing ? BSFixedWorkspace(n) : ws)
    end
end

"""
    bs_implied_vol_fixed_batch_threaded!(out, ks, cs, ::Val{N}, ::Val{W};
                                         vector_seed, ws)

Threads × SIMD: the batch kernel over contiguous per-thread chunks. Threads
write disjoint index ranges, so the two-pass workspace can be shared safely.
"""
function bs_implied_vol_fixed_batch_threaded!(out::Vector{Float64},
                                              ks::Vector{Float64}, cs::Vector{Float64},
                                              ::Val{N}, ::Val{W};
                                              vector_seed::Bool = (Sys.ARCH === :x86_64),
                                              ws::Union{Nothing,BSFixedWorkspace} = nothing) where {N,W}
    n = length(ks)
    @assert length(cs) == n && length(out) == n
    wsr = vector_seed ? nothing : (ws === nothing ? BSFixedWorkspace(n) : ws)
    nt = Threads.nthreads()
    chunk = cld(n, nt)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, n)
        lo > hi && continue
        if vector_seed
            i = lo
            @inbounds while i + W - 1 <= hi
                k = vload(Vec{W,Float64}, ks, i)
                c = vload(Vec{W,Float64}, cs, i)
                vstore(_solve_lanes(k, c, Val(N)), out, i)
                i += W
            end
            @inbounds while i <= hi
                out[i] = _solve_lanes(ks[i], cs[i], Val(N))
                i += 1
            end
        else
            @inbounds for i in lo:hi
                κ, cstar, E, invE = _otm_reduce(ks[i], cs[i])
                wsr.κ[i] = κ; wsr.cstar[i] = cstar; wsr.E[i] = E; wsr.invE[i] = invE
                out[i] = bs_seed(κ, cstar, E)
            end
            i = lo
            @inbounds while i + W - 1 <= hi
                v     = vload(Vec{W,Float64}, out, i)
                κ     = vload(Vec{W,Float64}, wsr.κ, i)
                cstar = vload(Vec{W,Float64}, wsr.cstar, i)
                E     = vload(Vec{W,Float64}, wsr.E, i)
                invE  = vload(Vec{W,Float64}, wsr.invE, i)
                for _ in 1:N
                    v = _hh4_update_bf(v, κ, cstar, E, invE)
                end
                vstore(v, out, i)
                i += W
            end
            @inbounds while i <= hi
                v = out[i]
                for _ in 1:N
                    v = _hh4_update_bf(v, wsr.κ[i], wsr.cstar[i], wsr.E[i], wsr.invE[i])
                end
                out[i] = v
                i += 1
            end
        end
    end
    return out
end
