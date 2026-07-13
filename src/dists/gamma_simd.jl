# Batched, SIMD-vectorized fixed-step Gamma quantile (fixed shape a, batch over
# u, the sampling workload), valid for a >= GAMMA_SIMD_AMIN.
#
# The incomplete-gamma CDF P(a,x) is normally iterative (series / continued
# fraction with data-dependent term counts), which forbids lane-uniform SIMD.
# Here it is replaced by Temme's uniform asymptotic expansion (DLMF 8.12) at
# FIXED order, so the whole residual is straight-line:
#
#   1/2 eta^2 = x/a - 1 - ln(x/a),   sign(eta) = sign(x/a - 1)
#   P(a,x) = Phi(eta*sqrt(a)) - e^{-a eta^2/2}/sqrt(2 pi a) * sum_{k=0}^6 c_k(eta)/a^k
#
# Each c_k(eta) is a fixed Taylor polynomial (graded degrees, generated in
# BigFloat); the eta->0 removable singularity is built into those Taylor forms,
# so no branch is needed near x = a. One vlog1p + one sqrt + two vexp + the
# blended Phi per residual. The log-space HH-4 polish is the same branch-free
# rational as the scalar gamma_log solver (F''/F' = a - x, F'''/F' = (a-x)^2 - x).
#
# a < GAMMA_SIMD_AMIN falls back to the scalar amortized batch: below ~20 the
# fixed-order Temme CDF cannot reach 1e-13 (see RESULTS.md).
using SIMD: Vec, vload, vstore
import SpecialFunctions: loggamma

const GAMMA_SIMD_AMIN = 20.0

# --- Temme c_k(eta) Taylor coefficients (auto-generated, BigFloat 300 bits),
# graded degrees (24,22,20,16,14,12,10); index 1 = eta^0, evalpoly convention. ---
const _TG_C0 = (-0.3333333333333333, 0.08333333333333333, -0.014814814814814815, 0.0011574074074074073, 0.0003527336860670194, -0.0001787551440329218, 3.919263178522438e-5, -2.185448510679992e-6, -1.85406221071516e-6, 8.296711340953087e-7, -1.7665952736826078e-7, 6.707853543401498e-9, 1.0261809784240309e-8, -4.382036018453353e-9, 9.14769958223679e-10, -2.5514193994946248e-11, -5.830772132550426e-11, 2.4361948020667415e-11, -5.0276692801141755e-12, 1.1004392031956135e-13, 3.371763262400985e-13, -1.392388722418162e-13, 2.8534893807047445e-14, -5.139111834242572e-16, -1.9752288294349442e-15)
const _TG_C1 = (-0.001851851851851852, -0.003472222222222222, 0.0026455026455026454, -0.0009902263374485596, 0.00020576131687242798, -4.018775720164609e-7, -1.8098550334489977e-5, 7.64916091608111e-6, -1.6120900894563446e-6, 4.647127802807434e-9, 1.378633446915721e-7, -5.752545603517705e-8, 1.1951628599778148e-8, -1.7543241719747647e-11, -1.0091543710600413e-9, 4.162792991842583e-10, -8.56390702649298e-11, 6.067215101604758e-14, 7.1624989648114856e-12, -2.933186643771437e-12, 5.996696365683689e-13, -2.1671786527323313e-16, -4.978339972369262e-14)
const _TG_C2 = (0.004133597883597883, -0.0026813271604938273, 0.0007716049382716049, 2.0093878600823047e-6, -0.0001073665322636516, 5.2923448829120125e-5, -1.2760635188618728e-5, 3.423578734096138e-8, 1.3721957309062934e-6, -6.298992138380055e-7, 1.4280614206064242e-7, -2.0477098421990866e-10, -1.409252991086752e-8, 6.228974084922022e-9, -1.3670488396617114e-9, 9.428356159014678e-13, 1.2872252400089318e-10, -5.5645956134363323e-11, 1.197593554636698e-11, -4.1689782251838634e-15, -1.0940640427884595e-12)
const _TG_C3 = (0.0006494341563786008, 0.00022947209362139917, -0.0004691894943952557, 0.00026772063206283885, -7.561801671883977e-5, -2.396505113867297e-7, 1.1082654115347302e-5, -5.6749528269915965e-6, 1.4230900732435883e-6, -2.7861080291528143e-11, -1.6958404091930278e-7, 8.099464905388083e-8, -1.9111168485973655e-8, 2.3928620439808118e-12, 2.0620131815488797e-9, -9.460496661855133e-10, 2.1541049775774907e-10)
const _TG_C4 = (-0.0008618882909167117, 0.0007840392217200666, -0.0002990724803031902, -1.4638452578843418e-6, 6.641498215465122e-5, -3.968365047179435e-5, 1.1375726970678419e-5, 2.507497226237533e-10, -1.6954149536558305e-6, 8.907507532205309e-7, -2.292934834000805e-7, 2.956794137544049e-11, 2.8865829742708783e-8, -1.4189739437803219e-8, 3.4463580499464896e-9)
const _TG_C5 = (-0.00033679855336635813, -6.972813758365857e-5, 0.0002772753244959392, -0.00019932570516188847, 6.797780477937208e-5, 1.419062920643967e-7, -1.3594048189768693e-5, 8.018470256334202e-6, -2.291481176508095e-6, -3.252473551298454e-10, 3.4652846491085265e-7, -1.8447187191171344e-7, 4.8240967037894184e-8)
const _TG_C6 = (0.0005313079364639922, -0.0005921664373536939, 0.0002708782096718045, 7.902353232660328e-7, -8.153969367561969e-5, 5.61168275310625e-5, -1.8329116582843375e-5, -3.0796134506033047e-9, 3.465155368803609e-6, -2.0291327396058603e-6, 5.788792863149004e-7)
# atanh series 1/(2j+1) for the accurate branch-free log1p near mu = 0
const _TG_ATANH = (1.0, 0.3333333333333333, 0.2, 0.14285714285714285, 0.1111111111111111, 0.09090909090909091, 0.07692307692307693, 0.06666666666666667, 0.058823529411764705, 0.05263157894736842, 0.047619047619047616)
# 11-term mantissa poly for a ~1e-16 branch-free log (vlog is only ~5e-13)
const _TG_VLOGC = (1.0, 0.3333333333333333, 0.2, 0.14285714285714285, 0.1111111111111111, 0.09090909090909091, 0.07692307692307693, 0.06666666666666667, 0.058823529411764705, 0.05263157894736842, 0.047619047619047616)

# accurate branch-free ln (11-term), reusing vlog's Cody-Waite reduction
@inline function _vlog_hi(x)
    xc = min(max(x, 1e-300), 1e300)
    ix = _bits(xc)
    ki = (ix - reinterpret(Int64, _VLOG_OFF)) >> 52
    m = _float(ix - (ki << 52))
    kf = _tofloat(ki)
    s = (m - 1.0) / (m + 1.0); w = s * s
    lm = 2.0 * s * evalpoly(w, _TG_VLOGC)
    return muladd(kf, _VLOG_LN2HI, lm) + kf * _VLOG_LN2LO
end

# accurate branch-free log1p: atanh series near 0 (no 1+mu rounding, exact for
# the eta cancellation), accurate reduced log in the tails.
@inline function _vlog1p(mu)
    r = mu / (2.0 + mu); w = r * r
    inner = 2.0 * r * evalpoly(w, _TG_ATANH)
    return sel(abs(mu) < 0.3, inner, _vlog_hi(1.0 + mu))
end

"Per-shape amortized constants for the Temme SIMD gamma kernel (a >= a_min)."
struct GammaTemmeQ
    a::Float64
    inva::Float64        # 1/a
    sqrta::Float64       # sqrt(a)
    cfp::Float64         # F' = cfp * g (g = e^{-a eta^2/2}); cfp = a^a e^{-a}/Gamma(a)
    inv_sqrt2pia::Float64 # 1/sqrt(2 pi a)
    lna::Float64         # ln a
end
function GammaTemmeQ(a::Float64)
    lga = loggamma(a)
    GammaTemmeQ(a, 1.0/a, sqrt(a), exp(a*log(a) - a - lga),
                1.0/sqrt(2.0*pi*a), log(a))
end

# Temme CDF residual and F' at y = ln x, on a scalar or a Vec lane-bundle.
# Returns (f = P - u, fp = F', x).
@inline function _temme_terms(D::GammaTemmeQ, y, u)
    x = vexp(y)
    mu = x * D.inva - 1.0
    half = mu - _vlog1p(mu)               # 1/2 eta^2 >= 0
    Q = D.a * half
    sq = sqrt(2.0 * max(half, 0.0))
    eta = sel(mu < 0.0, -sq, sq)
    g = vexp(-Q)
    Phi = phi_withg_bf(eta * D.sqrta, g)  # Phi(eta sqrt a); g = e^{-(eta sqrt a)^2/2}
    ia = D.inva
    S0 = evalpoly(eta, _TG_C0); S1 = evalpoly(eta, _TG_C1); S2 = evalpoly(eta, _TG_C2)
    S3 = evalpoly(eta, _TG_C3); S4 = evalpoly(eta, _TG_C4); S5 = evalpoly(eta, _TG_C5)
    S6 = evalpoly(eta, _TG_C6)
    Ssum = S0 + ia*(S1 + ia*(S2 + ia*(S3 + ia*(S4 + ia*(S5 + ia*S6)))))
    P = Phi - g * D.inv_sqrt2pia * Ssum
    return P - u, D.cfp * g, x
end

# lane-uniform seed y = ln x0: Cornish-Fisher (ODE5) blended with Wilson-Hilferty
# (both branch-free), selected by tail depth; WH also guards non-finite CF.
@inline function _gamma_temme_seed(D::GammaTemmeQ, u)
    z = norminv_bf(u)
    a = D.a; sa = D.sqrta
    # Wilson-Hilferty cube
    tw = 1.0 - 1.0/(9.0*a) + z/(3.0*sa)
    xwh = a * tw * tw * tw
    # 5th-order Cornish-Fisher in eps = 1/sqrt(a)
    eps = 1.0/sa; z2 = z*z
    p1 = (z2 - 1.0)/3.0
    p2 = z*(z2 - 7.0)/36.0
    p3 = -(3.0*z2*z2 + 7.0*z2 - 16.0)/810.0
    p4 = z*(9.0*z2*z2 + 256.0*z2 - 433.0)/38880.0
    p5 = (12.0*z2*z2*z2 - 243.0*z2*z2 - 923.0*z2 + 1472.0)/204120.0
    ycf = z + eps*(p1 + eps*(p2 + eps*(p3 + eps*(p4 + eps*p5))))
    xcf = a + sa*ycf
    x0 = sel((abs(z) <= 3.5) & (xcf > 0.0), xcf, xwh)
    x0 = sel(x0 > 0.0, x0, 1e-8)
    return _vlog_hi(x0)
end

# reduce -> seed -> N HH-4 updates -> x, on one scalar or one Vec bundle
@inline function _gamma_temme_solve_lanes(D::GammaTemmeQ, u, ::Val{N}) where {N}
    y = _gamma_temme_seed(D, u)
    a = D.a
    for _ in 1:N
        f, fp, x = _temme_terms(D, y, u)
        r = f / fp
        φ2 = a - x
        ξ = φ2 * φ2 - x
        denom = -6.0 + r * (6.0 * φ2 - r * ξ)
        yn = y + 3.0 * r * (2.0 - r * φ2) / denom
        yn = sel(abs(denom) < 1e-20, y - r, yn)
        yn = sel(isfinite(yn), yn, y)
        y = min(max(yn, -746.0), 710.0)
    end
    x = vexp(y)
    x = sel(u <= 0.0, 0.0, sel(u >= 1.0, Inf, x))
    return x
end

"""
    gamma_quantile_batch_simd!(out, a, us, ::Val{N}, ::Val{W})

Batched Gamma(shape `a`, scale 1) quantiles at fixed `a`: lane-uniform
Cornish-Fisher/Wilson-Hilferty seed + exactly `N` HH-4 updates whose CDF is the
fixed-order Temme uniform asymptotic expansion, vectorized `W` lanes wide.
Valid for `a >= GAMMA_SIMD_AMIN` (= $(GAMMA_SIMD_AMIN)); smaller `a` delegates to
the scalar [`gamma_quantile_batch!`]. Allocation-free given a preallocated `out`.
"""
function gamma_quantile_batch_simd!(out::Vector{Float64}, a::Float64,
                                    us::Vector{Float64}, ::Val{N}, ::Val{W}) where {N,W}
    n = length(us)
    length(out) == n || throw(DimensionMismatch("out and us must have equal length"))
    if a < GAMMA_SIMD_AMIN
        return gamma_quantile_batch!(out, a, us)
    end
    D = GammaTemmeQ(a)
    i = 1
    @inbounds while i + W - 1 <= n
        u = vload(Vec{W,Float64}, us, i)
        vstore(_gamma_temme_solve_lanes(D, u, Val(N)), out, i)
        i += W
    end
    @inbounds while i <= n
        out[i] = _gamma_temme_solve_lanes(D, us[i], Val(N))
        i += 1
    end
    return out
end

"Threads x SIMD variant of [`gamma_quantile_batch_simd!`] (contiguous chunks)."
function gamma_quantile_batch_simd_threaded!(out::Vector{Float64}, a::Float64,
                                             us::Vector{Float64}, ::Val{N}, ::Val{W}) where {N,W}
    n = length(us)
    length(out) == n || throw(DimensionMismatch("out and us must have equal length"))
    if a < GAMMA_SIMD_AMIN
        return gamma_quantile_batch!(out, a, us)
    end
    D = GammaTemmeQ(a)
    nt = Threads.nthreads()
    chunk = cld(n, nt)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, n)
        lo > hi && continue
        i = lo
        @inbounds while i + W - 1 <= hi
            u = vload(Vec{W,Float64}, us, i)
            vstore(_gamma_temme_solve_lanes(D, u, Val(N)), out, i)
            i += W
        end
        @inbounds while i <= hi
            out[i] = _gamma_temme_solve_lanes(D, us[i], Val(N))
            i += 1
        end
    end
    return out
end
