# Beta ODE5 central seed + y6 error certificate, ported from A. Hekimoglu's beta
# engine (his ODE5 central z-polynomial seed and its seed-intrinsic y6 term).
#
# In the high-concentration central region we write the quantile in standardized
# coordinates: X = p + eps*sqrt(pq)*Y with p = a/n, n = a+b, eps = n^(-1/2), and
# expand Y in powers of eps against the standard-normal deviate z = Phi^{-1}(u):
#   Y = z + eps*y1(z) + eps^2*y2(z) + ... + eps^5*y5(z).
# The coefficient polynomials y_k(z) are fixed per shape (they depend only on p),
# computed once by solving the quantile ODE order by order from the beta
# log-density expansion. y_k has degree k+1; the omitted eps^6*y6(z) term gives
# the leading seed-error estimate used by the certificate.
#
# The raw seed is an asymptotic-in-n object: exact as n -> Inf, so its logit
# accuracy runs ~1e-3 at n~5 to ~1e-8 at n~200 (see RESULTS). To reach 1e-14 we
# feed it into the existing logit-HH4 certified solver: one CDF evaluation gives
# the true residual r = f/F', the classical Householder-3 error model K4*r^4
# certifies the single update, and anything uncertified falls back bit-for-bit to
# the standard solver. The sharper ODE5 seed lifts certificate coverage far above
# the CF5 seed at small/moderate n, which is where the speed comes from.
import SpecialFunctions: logbeta

# -----------------------------------------------------------------------------
# Polynomial series algebra (isbits, allocation-free). A degree-<=BP_N-1
# polynomial in z is an NTuple{BP_N,Float64} with coefficient of z^i at slot i+1.
# The ODE5 build peaks at degree 8, so BP_N = 10 leaves margin.
# -----------------------------------------------------------------------------
const BP_N = 10
const BP_ZERO = ntuple(_ -> 0.0, Val(BP_N))
const BSERIES_ZERO = ntuple(_ -> BP_ZERO, Val(7))   # eps-orders 0..6

@inline bp_setcoef(p::NTuple{BP_N,Float64}, deg::Int, v::Float64) = Base.setindex(p, v, deg + 1)
@inline bp_add(a::NTuple{BP_N,Float64}, b::NTuple{BP_N,Float64}) = ntuple(i -> a[i] + b[i], Val(BP_N))
@inline bp_sub(a::NTuple{BP_N,Float64}, b::NTuple{BP_N,Float64}) = ntuple(i -> a[i] - b[i], Val(BP_N))
@inline bp_scale(a::NTuple{BP_N,Float64}, s::Float64) = ntuple(i -> s * a[i], Val(BP_N))
@inline function bp_mul(a::NTuple{BP_N,Float64}, b::NTuple{BP_N,Float64})
    ntuple(Val(BP_N)) do i
        s = 0.0
        @inbounds for j in 1:i
            s += a[j] * b[i-j+1]
        end
        s
    end
end
@inline function bp_eval(p::NTuple{BP_N,Float64}, z::Float64)
    r = 0.0
    @inbounds for i in BP_N:-1:1
        r = r * z + p[i]
    end
    return r
end
@inline bp_e0() = ntuple(i -> i == 1 ? 1.0 : 0.0, Val(BP_N))   # the constant 1
@inline bp_z() = ntuple(i -> i == 2 ? 1.0 : 0.0, Val(BP_N))    # the monomial z

const BSeries = NTuple{7,NTuple{BP_N,Float64}}

@inline function bseries_mul(A::BSeries, B::BSeries, maxord::Int)
    C = BSERIES_ZERO
    for i in 0:maxord, j in 0:(maxord-i)
        idx = i + j + 1
        C = Base.setindex(C, bp_add(C[idx], bp_mul(A[i+1], B[j+1])), idx)
    end
    C
end
@inline function bseries_exp(D::BSeries, maxord::Int)
    R = Base.setindex(BSERIES_ZERO, bp_e0(), 1)
    for k in 1:maxord
        s = BP_ZERO
        for i in 1:k
            s = bp_add(s, bp_scale(bp_mul(D[i+1], R[k-i+1]), Float64(i)))
        end
        R = Base.setindex(R, bp_scale(s, 1.0 / k), k + 1)
    end
    R
end
# Composition sum_m P_m Y^m as an eps-series (P a polynomial in the base var).
@inline function bpoly_compose_series(P::NTuple{BP_N,Float64}, Y::BSeries, maxord::Int)
    pdeg = BP_N - 1
    while pdeg > 0 && P[pdeg+1] == 0.0
        pdeg -= 1
    end
    OUT = BSERIES_ZERO
    power = Base.setindex(BSERIES_ZERO, bp_e0(), 1)
    for m in 0:pdeg
        m > 0 && (power = bseries_mul(power, Y, maxord))
        pc = P[m+1]
        if pc != 0.0
            for k in 0:maxord
                OUT = Base.setindex(OUT, bp_add(OUT[k+1], bp_scale(power[k+1], pc)), k + 1)
            end
        end
    end
    OUT
end

# The z-polynomials L1..L6 of the standardized beta log-density expansion.
@inline function beta_logdensity_L_polys(p::Float64)
    c = sqrt(max(p * (1.0 - p), floatmin(Float64)))
    p2 = p * p; p3 = p2 * p; p4 = p2 * p2; pm = p - 1.0; qv = 1.0 - p
    L1 = BP_ZERO; L2 = BP_ZERO; L3 = BP_ZERO; L4 = BP_ZERO; L5 = BP_ZERO; L6 = BP_ZERO
    k = (1.0 - 2.0 * p) / (3.0 * c)
    L1 = bp_setcoef(L1, 1, -3.0 * k); L1 = bp_setcoef(L1, 3, k)
    den2 = 12.0 * p * pm
    L2 = bp_setcoef(L2, 0, (p2 - p + 1.0) / den2)
    L2 = bp_setcoef(L2, 2, (-12.0 * p2 + 12.0 * p - 6.0) / den2)
    L2 = bp_setcoef(L2, 4, (9.0 * p2 - 9.0 * p + 3.0) / den2)
    den3 = 15.0 * p2 * pm * pm
    A3 = 6.0 * p2 - 6.0 * p + 3.0; B3 = -5.0 * p2 + 5.0 * p - 5.0
    L3 = bp_setcoef(L3, 5, -c * (2.0 * p - 1.0) * A3 / den3)
    L3 = bp_setcoef(L3, 3, -c * (2.0 * p - 1.0) * B3 / den3)
    den4 = 12.0 * p2 * pm * pm
    A4 = 10.0 * p4 - 20.0 * p3 + 20.0 * p2 - 10.0 * p + 2.0
    B4 = -6.0 * p4 + 12.0 * p3 - 18.0 * p2 + 12.0 * p - 3.0
    L4 = bp_setcoef(L4, 6, -A4 / den4); L4 = bp_setcoef(L4, 4, -B4 / den4)
    den5 = 35.0 * p2 * p * pm * pm * pm
    A5 = 15.0 * p4 - 30.0 * p3 + 35.0 * p2 - 20.0 * p + 5.0
    B5 = -7.0 * p4 + 14.0 * p3 - 28.0 * p2 + 21.0 * p - 7.0
    L5 = bp_setcoef(L5, 7, c * (2.0 * p - 1.0) * A5 / den5)
    L5 = bp_setcoef(L5, 5, c * (2.0 * p - 1.0) * B5 / den5)
    c6p = c^6.0; c8p = c6p * c * c
    L6 = bp_setcoef(L6, 6, (c6p / 6.0) * (1.0 / p^6.0 + 1.0 / qv^6.0))
    L6 = bp_setcoef(L6, 8, -(c8p / 8.0) * (1.0 / p^7.0 + 1.0 / qv^7.0))
    (L1, L2, L3, L4, L5, L6)
end

# Solve L[y] = y' - z y = K for the polynomial y, via the coefficient recurrence
# K_j = (j+1) a_{j+1} - a_{j-1}.  Returns y with deg y = deg K - 1.
@inline function solve_L_poly(K::NTuple{BP_N,Float64})
    m = BP_N - 1
    while m > 0 && K[m+1] == 0.0
        m -= 1
    end
    m <= 0 && return BP_ZERO
    D = m - 1
    acoef = BP_ZERO
    for j in (D+1):-1:1
        Kj = j <= BP_N - 1 ? K[j+1] : 0.0
        ap1 = (j + 1 <= D) ? acoef[j+1+1] : 0.0
        acoef = Base.setindex(acoef, (j + 1) * ap1 - Kj, j - 1 + 1)
    end
    acoef
end

# Build the six seed coefficient polynomials y1..y6 for shape parameter p = a/n.
@inline function beta_ode5_ycoef(p::Float64)
    Ls = beta_logdensity_L_polys(p)
    Yser = Base.setindex(BSERIES_ZERO, bp_z(), 1)   # Y_0 = z
    z2 = bp_mul(bp_z(), bp_z())
    Yc = ntuple(_ -> BP_ZERO, Val(6))
    for k in 1:6
        Y2 = bseries_mul(Yser, Yser, k)
        Dser = Base.setindex(BSERIES_ZERO, bp_scale(bp_sub(Y2[1], z2), 0.5), 1)
        for j in 1:k
            Dser = Base.setindex(Dser, bp_scale(Y2[j+1], 0.5), j + 1)
        end
        for m in 1:min(k, 6)
            comp = bpoly_compose_series(Ls[m], Yser, k - m)
            for j in 0:(k-m)
                Dser = Base.setindex(Dser, bp_sub(Dser[m+j+1], comp[j+1]), m + j + 1)
            end
        end
        R = bseries_exp(Dser, k)
        yk = solve_L_poly(R[k+1])
        Yser = Base.setindex(Yser, yk, k + 1)
        Yc = Base.setindex(Yc, yk, k)
    end
    Yc
end

# -----------------------------------------------------------------------------
# Per-shape ODE5 cache (isbits, so the constructor is allocation-free).
# -----------------------------------------------------------------------------
struct BetaODE5
    p::Float64      # a/n
    c::Float64      # sqrt(p*(1-p))
    eps::Float64    # 1/sqrt(n)
    Y::NTuple{6,NTuple{BP_N,Float64}}   # y1..y6
end
function BetaODE5(a::Float64, b::Float64)
    n = a + b; p = a / n; q = 1.0 - p
    BetaODE5(p, sqrt(max(p * q, floatmin(Float64))), 1.0 / sqrt(n), beta_ode5_ycoef(p))
end

# Shape qualifies for the central ODE5 treatment (matches Hekimoglu's gate).
@inline beta_ode5_shape_ok(a, b) = (n = a + b; p = a / n; n >= 4.0 && 0.05 <= p <= 0.95)

# Raw seed x0 = p + eps*sqrt(pq)*(z + eps y1 + ... + eps^5 y5), z = Phi^{-1}(u).
@inline function beta_ode5_seed_x(D::BetaODE5, z::Float64)
    Y = z; ep = D.eps
    @inbounds for k in 1:5
        Y += ep * bp_eval(D.Y[k], z)
        ep *= D.eps
    end
    return D.p + D.eps * D.c * Y
end

# Leading seed-error estimate in logit units: the omitted eps^6 y6 term mapped
# through dlogit = dx/(x(1-x)).  e0 = |c eps^7 y6(z) / (x0 (1-x0))|.
@inline function beta_ode5_seed_e0(D::BetaODE5, z::Float64, x0::Float64)
    y6 = bp_eval(D.Y[6], z)
    eps2 = D.eps * D.eps
    eps7 = eps2 * eps2 * eps2 * D.eps
    abs(D.c * eps7 * y6 / (x0 * (1.0 - x0)))
end

"""
    beta_ode5_seed(a, b, u)

Raw ODE5 central seed for the Beta(a,b) quantile, using ZERO incomplete-beta
evaluations. This is an asymptotic-in-n object (exact as n -> Inf): its logit
accuracy is roughly 1e-3 at n~5 and 1e-8 at n~200, NOT machine precision. It is
Hekimoglu's fast Monte-Carlo / calibration tier; use [`beta_quantile_ode5`] for a
certified 1e-14 quantile. Outside the central gate it defers to
[`beta_quantile_logit`].
"""
@inline function beta_ode5_seed(a::Float64, b::Float64, u::Float64)
    u <= 0.0 && return 0.0
    u >= 1.0 && return 1.0
    (b == 1.0 || a == 1.0 || (a == 0.5 && b == 0.5) || !beta_ode5_shape_ok(a, b)) &&
        return beta_quantile_logit(a, b, u)
    z = norminv(u)
    if abs(z) <= 2.5
        x0 = beta_ode5_seed_x(BetaODE5(a, b), z)
        (0.0 < x0 < 1.0) && return x0
    end
    return beta_quantile_logit(a, b, u)
end

# Certified fast step: ODE5 seed + ONE logit-HH4 update, gated by the classical
# Householder-3 error model K4*r^4 on the true residual r = f/F' (one CDF eval).
# Returns the certified logit value, or NaN when the point is not certified (the
# caller then takes the exact standard-solver path, keeping the result outside
# the certified region bit-for-bit identical to `beta_quantile_logit`).
@inline function beta_ode5_cert_logit(D::BetaODE5, Dlg::BetaLogitQ, u::Float64,
                                      tol::Float64, τ::Float64, safety::Float64)
    z = norminv(u)
    abs(z) <= 2.5 || return NaN
    x0 = beta_ode5_seed_x(D, z)
    (0.0 < x0 < 1.0) || return NaN
    y0 = log(x0) - log1p(-x0)
    f, fp, φ2, ξ = hh_terms(Dlg, y0, u)
    converged(Dlg, f, fp, tol) && return y0
    r = f / fp
    c2 = 0.5 * φ2
    c3 = ξ / 6.0
    c4 = hh4_c4(Dlg, y0) / 24.0
    K4 = abs(5.0 * c2 * (c2 * c2 - c3) + c4)
    r2 = r * r
    safety * K4 * r2 * r2 <= τ || return NaN
    denom = -6.0 + r * (6.0 * φ2 - r * ξ)
    return abs(denom) < 1e-20 ? y0 - r : y0 + 3.0 * r * (2.0 - r * φ2) / denom
end

"""
    beta_quantile_ode5(a, b, u; tol=1e-14, τ=1e-14, safety=16.0)

Beta(a,b) quantile using the ODE5 central seed. In the high-concentration central
region the sharp seed lets the K4 certificate accept a single logit-HH-4 update
(one incomplete-beta evaluation); everywhere else, and on any point the
certificate declines, it is bit-for-bit [`beta_quantile_logit`]. Certified points
match the full solver to <= ~3e-15 in logit.
"""
@inline function beta_quantile_ode5(a::Float64, b::Float64, u::Float64;
                                    tol::Float64 = 1e-14, τ::Float64 = 1e-14, safety::Float64 = 16.0)
    u <= 0.0 && return 0.0
    u >= 1.0 && return 1.0
    (b == 1.0 || a == 1.0 || (a == 0.5 && b == 0.5) || !beta_ode5_shape_ok(a, b)) &&
        return beta_quantile_logit(a, b, u; tol = tol)
    yn = beta_ode5_cert_logit(BetaODE5(a, b), BetaLogitQ(a, b), u, tol, τ, safety)
    isnan(yn) && return beta_quantile_logit(a, b, u; tol = tol)
    return 1.0 / (1.0 + exp(-yn))
end

"""
    beta_quantile_ode5_batch!(out, a, b, us; tol=1e-14, τ=1e-14, safety=16.0)

Amortized ODE5 batch: the per-shape seed polynomials (and the mirror
`BetaLogitQ`) are built once, then reused across `us`. Certified central points
take the one-evaluation fast path; every other point is bit-for-bit
[`beta_quantile_batch!`]`(certified=true)`. Allocation-free given preallocated
`out`.
"""
function beta_quantile_ode5_batch!(out::Vector{Float64}, a::Float64, b::Float64, us::Vector{Float64};
                                   tol::Float64 = 1e-14, τ::Float64 = 1e-14, safety::Float64 = 16.0)
    length(out) == length(us) || throw(DimensionMismatch("out and us must have equal length"))
    # exact-form shapes and non-central shapes: no ODE5 benefit, defer to the
    # standard certified batch (identical results, one code path).
    if b == 1.0 || a == 1.0 || (a == 0.5 && b == 0.5) || !beta_ode5_shape_ok(a, b)
        return beta_quantile_batch!(out, a, b, us; tol = tol, certified = true)
    end
    Dab = BetaLogitQ(a, b)     # central + fallback (u <= 1/2)
    Dba = BetaLogitQ(b, a)     # mirrored fallback (u > 1/2)
    D5 = BetaODE5(a, b)
    @inbounds for i in eachindex(us)
        u = us[i]
        if u <= 0.0
            out[i] = 0.0
        elseif u >= 1.0
            out[i] = 1.0
        else
            yn = beta_ode5_cert_logit(D5, Dab, u, tol, τ, safety)
            if !isnan(yn)
                out[i] = 1.0 / (1.0 + exp(-yn))          # certified fast path
            elseif u <= 0.5
                y = solve_certified(Dab, u; tol = tol)   # exact fallback == main
                out[i] = 1.0 / (1.0 + exp(-y))
            else
                y = solve_certified(Dba, 1.0 - u; tol = tol)
                ey = exp(y)
                out[i] = 1.0 - ey / (1.0 + ey)
            end
        end
    end
    return out
end

"""
    beta_ode5_seed_batch!(out, a, b, us)

Amortized raw ODE5 seed batch (ZERO incomplete-beta evaluations): builds the seed
polynomials once, then fills `out` with the raw central seed for every `u` in the
gate. Approximate / asymptotic-in-n (see [`beta_ode5_seed`]) - for Monte-Carlo and
calibration workloads that need the sampled distribution right, not each quantile
to machine precision. Off-gate points defer to the exact standard path.
Allocation-free given preallocated `out`.
"""
function beta_ode5_seed_batch!(out::Vector{Float64}, a::Float64, b::Float64, us::Vector{Float64})
    length(out) == length(us) || throw(DimensionMismatch("out and us must have equal length"))
    if b == 1.0 || a == 1.0 || (a == 0.5 && b == 0.5) || !beta_ode5_shape_ok(a, b)
        return beta_quantile_batch!(out, a, b, us; certified = true)
    end
    Dab = BetaLogitQ(a, b); Dba = BetaLogitQ(b, a)
    D5 = BetaODE5(a, b)
    @inbounds for i in eachindex(us)
        u = us[i]
        if u <= 0.0
            out[i] = 0.0
        elseif u >= 1.0
            out[i] = 1.0
        else
            z = norminv(u)
            done = false
            if abs(z) <= 2.5
                x0 = beta_ode5_seed_x(D5, z)
                if 0.0 < x0 < 1.0
                    out[i] = x0; done = true
                end
            end
            if !done
                if u <= 0.5
                    y = solve_certified(Dab, u)
                    out[i] = 1.0 / (1.0 + exp(-y))
                else
                    y = solve_certified(Dba, 1.0 - u)
                    ey = exp(y); out[i] = 1.0 - ey / (1.0 + ey)
                end
            end
        end
    end
    return out
end

"Mode oracle for coverage measurement: 1 if u is on the certified ODE5 fast
path for Beta(a,b), else 0."
@inline function beta_quantile_ode5_mode(a::Float64, b::Float64, u::Float64;
                                         tol::Float64 = 1e-14, τ::Float64 = 1e-14, safety::Float64 = 16.0)
    (u <= 0.0 || u >= 1.0) && return 0
    (b == 1.0 || a == 1.0 || (a == 0.5 && b == 0.5) || !beta_ode5_shape_ok(a, b)) && return 0
    isnan(beta_ode5_cert_logit(BetaODE5(a, b), BetaLogitQ(a, b), u, tol, τ, safety)) ? 0 : 1
end
