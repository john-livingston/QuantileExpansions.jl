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

# --- branch-free natural log (for the blended Acklam tails) ------------------
# Exponent bit-extract + atanh-series mantissa polynomial. Absolute error
# ~5e-13 — far more than the seeds need (their norminv tolerance is ~1e-6;
# the HH-4 polish erases seed-level differences quartically).
const _VLOG_OFF = 0x3fe6a09e667f3bcd            # bits of √0.5: mantissa ∈ [√0.5, √2)
const _VLOG_LN2HI = 0.6931471803691238
const _VLOG_LN2LO = 1.9082149292705877e-10
# 2·atanh(s) = 2s·(1 + w/3 + w²/5 + …), w = s²; |s| ≤ 0.1716 ⇒ w ≤ 0.0295
const _VLOG_C = (1.0, 1.0/3, 1.0/5, 1.0/7, 1.0/9, 1.0/11, 1.0/13)

@inline _bits(x::Float64) = reinterpret(Int64, x)
@inline _bits(x::Vec{W,Float64}) where {W} = reinterpret(Vec{W,Int64}, x)
@inline _float(i::Int64) = reinterpret(Float64, i)
@inline _float(i::Vec{W,Int64}) where {W} = reinterpret(Vec{W,Float64}, i)
@inline _tofloat(i::Int64) = Float64(i)
@inline _tofloat(i::Vec{W,Int64}) where {W} = convert(Vec{W,Float64}, i)

@inline function vlog(x)
    xc = min(max(x, 1e-300), 1e300)             # keep strictly positive/finite
    ix = _bits(xc)
    ki = (ix - reinterpret(Int64, _VLOG_OFF)) >> 52
    m = _float(ix - (ki << 52))                 # mantissa ∈ [√0.5, √2)
    kf = _tofloat(ki)
    s = (m - 1.0) / (m + 1.0)
    w = s * s
    lm = 2.0 * s * evalpoly(w, _VLOG_C)
    return muladd(kf, _VLOG_LN2HI, lm) + kf * _VLOG_LN2LO
end

# --- branch-free Acklam Φ⁻¹ (all three branches evaluated, lane-selected) -----
# Same rationals as norminv in specialfuns.jl; tails use q = √(-2·vlog(p̃)).
# Out-of-domain p ≤ 0 / ≥ 1 saturates to ∓38 like the scalar guard.
@inline function norminv_bf(p)
    pc = min(max(p, 1e-300), 1.0 - 1.1e-16)
    # central branch
    q = pc - 0.5
    r = q * q
    zc = (((((_AK_A[1]*r+_AK_A[2])*r+_AK_A[3])*r+_AK_A[4])*r+_AK_A[5])*r+_AK_A[6])*q /
         (((((_AK_B[1]*r+_AK_B[2])*r+_AK_B[3])*r+_AK_B[4])*r+_AK_B[5])*r+1.0)
    # tail branch on the smaller side, sign-folded
    pt = min(pc, 1.0 - pc)
    qt = sqrt(-2.0 * vlog(pt))
    zt = (((((_AK_C[1]*qt+_AK_C[2])*qt+_AK_C[3])*qt+_AK_C[4])*qt+_AK_C[5])*qt+_AK_C[6]) /
         ((((_AK_D[1]*qt+_AK_D[2])*qt+_AK_D[3])*qt+_AK_D[4])*qt+1.0)
    zt = sel(pc < 0.5, zt, -zt)
    z = sel((pc >= 0.02425) & (pc <= 0.97575), zc, zt)
    return min(max(z, -38.0), 38.0)
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
