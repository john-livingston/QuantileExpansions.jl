# Branch-free math generic over Float64 and SIMD.Vec{W,Float64}.
#
# The HH-4 polish iteration is branch-free at the *algorithm* level (fixed step
# count, select-based guards) but the scalar special functions still branch on
# range. Everything here trades those branches for blended evaluation + select,
# so the whole polish body compiles to straight-line vector code.

using SIMD: Vec, vifelse

# select: Base.ifelse for scalars, per-lane vifelse for Vec masks
@inline sel(m::Bool, a, b) = ifelse(m, a, b)
@inline sel(m::Vec{W,Bool}, a, b) where {W} = vifelse(m, a, b)

# 2^n from an integral-valued float n via the exponent-field bit trick
@inline function _pow2(n::Float64)
    reinterpret(Float64, (unsafe_trunc(Int64, n) + 1023) << 52)
end
@inline function _pow2(n::Vec{W,Float64}) where {W}
    reinterpret(Vec{W,Float64}, (convert(Vec{W,Int64}, n) + 1023) << 52)
end

const _VEXP_LOG2E = 1.4426950408889634
const _VEXP_LN2HI = 0.6931471803691238    # high part of ln2 (Cody–Waite)
const _VEXP_LN2LO = 1.9082149292705877e-10
# Taylor coefficients 1/k!, k = 0..13; |r| ≤ ln2/2 ⇒ truncation < 2e-16 relative
const _VEXP_C = (1.0, 1.0, 0.5, 1.0/6, 1.0/24, 1.0/120, 1.0/720, 1.0/5040,
                 1.0/40320, 1.0/362880, 1.0/3628800, 1.0/39916800,
                 1.0/479001600, 1.0/6227020800)

# exp(x), branch-free, ~1 ulp: Cody–Waite range reduction + degree-13 poly +
# exponent bit-trick scaling. Arguments are clamped to ±708 (no overflow, and
# results that would underflow return ~4e-308 instead of 0 — acceptable for the
# polish, whose gaussian factors stay ≥ e^{-800} only when |d| is far outside
# the solver's validity domain).
@inline function vexp(x)
    xc = min(max(x, -708.0), 708.0)
    n = round(xc * _VEXP_LOG2E)
    r = muladd(n, -_VEXP_LN2HI, xc)
    r = muladd(n, -_VEXP_LN2LO, r)
    return evalpoly(r, _VEXP_C) * _pow2(n)
end

# --- blended Cody Φ (all three range branches evaluated, lane-selected) -------
# Same rationals as specialfuns.jl; the small branch is the erf rational, the
# mid/large are the erfcx rationals scaled by the caller-supplied g = e^{-x²/2}.
@inline function phi_withg_bf(x, g)
    xe = -x * INV_SQRT2
    y = abs(xe)
    # small branch: erf rational (y ≤ 0.46875), no exp needed
    z = y * y
    num_s = _ERF_P[5] * z
    den_s = z
    num_s = (num_s + _ERF_P[1]) * z; den_s = (den_s + _ERF_Q[1]) * z
    num_s = (num_s + _ERF_P[2]) * z; den_s = (den_s + _ERF_Q[2]) * z
    num_s = (num_s + _ERF_P[3]) * z; den_s = (den_s + _ERF_Q[3]) * z
    erf_s = xe * (num_s + _ERF_P[4]) / (den_s + _ERF_Q[4])
    Φ_s = 0.5 * (1.0 - erf_s)
    # mid branch: erfcx rational (0.46875 < y ≤ 4)
    num_m = _ERFC_P[9] * y
    den_m = y
    num_m = (num_m + _ERFC_P[1]) * y; den_m = (den_m + _ERFC_Q[1]) * y
    num_m = (num_m + _ERFC_P[2]) * y; den_m = (den_m + _ERFC_Q[2]) * y
    num_m = (num_m + _ERFC_P[3]) * y; den_m = (den_m + _ERFC_Q[3]) * y
    num_m = (num_m + _ERFC_P[4]) * y; den_m = (den_m + _ERFC_Q[4]) * y
    num_m = (num_m + _ERFC_P[5]) * y; den_m = (den_m + _ERFC_Q[5]) * y
    num_m = (num_m + _ERFC_P[6]) * y; den_m = (den_m + _ERFC_Q[6]) * y
    num_m = (num_m + _ERFC_P[7]) * y; den_m = (den_m + _ERFC_Q[7]) * y
    R_m = (num_m + _ERFC_P[8]) / (den_m + _ERFC_Q[8])
    # large branch: asymptotic erfcx rational (y > 4); guard y=0 division
    zz = 1.0 / max(z, 1e-300)
    num_l = _ERFC_R[6] * zz
    den_l = zz
    num_l = (num_l + _ERFC_R[1]) * zz; den_l = (den_l + _ERFC_S[1]) * zz
    num_l = (num_l + _ERFC_R[2]) * zz; den_l = (den_l + _ERFC_S[2]) * zz
    num_l = (num_l + _ERFC_R[3]) * zz; den_l = (den_l + _ERFC_S[3]) * zz
    num_l = (num_l + _ERFC_R[4]) * zz; den_l = (den_l + _ERFC_S[4]) * zz
    R_l = zz * (num_l + _ERFC_R[5]) / (den_l + _ERFC_S[5])
    R_l = (0.5641895835477563 - R_l) / max(y, 1e-300)
    # blend: ec = erfc(y)·(sign fold), Φ = ec/2 or 1 - ec/2
    ec = sel(y <= 4.0, R_m, R_l) * g
    Φ_ml = sel(xe < 0.0, 1.0 - 0.5 * ec, 0.5 * ec)
    return sel(y <= 0.46875, Φ_s, Φ_ml)
end

# Φ(x) and φ(x) sharing one branch-free exp
@inline function phi_cdf_pdf_bf(x)
    g = vexp(-0.5 * x * x)
    return phi_withg_bf(x, g), INV_SQRT2PI * g
end
