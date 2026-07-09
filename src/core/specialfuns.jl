# Special functions: normal CDF Φ, inverse normal CDF Φ⁻¹, erfc.
# Goal: fast + accurate enough that the HH-4 residual hits |f| < 1e-14.

const INV_SQRT2  = 0.7071067811865476      # 1/√2
const INV_SQRT2PI = 0.3989422804014327     # 1/√(2π)  == a1
const SQRT2PI    = 2.5066282746310002      # √(2π)

# --- Standard normal CDF via erfc -------------------------------------------
# Φ(x) = ½ erfc(-x/√2).  We rely on a high-accuracy erfc (below).
@inline normcdf(x::Float64) = 0.5 * erfc_hi(-x * INV_SQRT2)

# Standard normal pdf
@inline normpdf(x::Float64) = INV_SQRT2PI * exp(-0.5 * x * x)

# --- High-accuracy erfc (W. J. Cody, 1969 rational approximations) -----------
# This is the same algorithm Cephes / many C libms use, so benchmarking against
# it is a fair "beat C" comparison.  Relative accuracy ~1e-15.
const _ERF_P = (3.16112374387056560e00, 1.13864154151050156e02,
                3.77485237685302021e02, 3.20937758913846947e03,
                1.85777706184603153e-1)
const _ERF_Q = (2.36012909523441209e01, 2.44024637934444173e02,
                1.28261652607737228e03, 2.84423683343917062e03)
const _ERFC_P = (5.64188496988670089e-1, 8.88314979438837594e00,
                 6.61191906371416295e01, 2.98635138197400131e02,
                 8.81952221241769090e02, 1.71204761263407058e03,
                 2.05107837782607147e03, 1.23033935479799725e03,
                 2.15311535474403846e-8)
const _ERFC_Q = (1.57449261107098347e01, 1.17693950891312499e02,
                 5.37181101862009858e02, 1.62138957456669019e03,
                 3.29079923573345963e03, 4.36261909014324716e03,
                 3.43936767414372164e03, 1.23033935480374942e03)
const _ERFC_R = (3.05326634961232344e-1, 3.60344899949804439e-1,
                 1.25781726111229246e-1, 1.60837851487422766e-2,
                 6.58749161529837803e-4, 1.63153871373020978e-2)
const _ERFC_S = (2.56852019228982242e00, 1.87295284992346047e00,
                 5.27905102951428412e-1, 6.05183413124413191e-2,
                 2.33520497626869185e-3)

@inline function erfc_hi(x::Float64)
    y = abs(x)
    if y <= 0.46875
        # erf on [0, 0.46875]; erfc = 1 - erf
        z = y * y
        num = _ERF_P[5] * z
        den = z
        @inbounds for i in 1:3
            num = (num + _ERF_P[i]) * z
            den = (den + _ERF_Q[i]) * z
        end
        r = x * (num + _ERF_P[4]) / (den + _ERF_Q[4])
        return 1.0 - r
    elseif y <= 4.0
        num = _ERFC_P[9] * y
        den = y
        @inbounds for i in 1:7
            num = (num + _ERFC_P[i]) * y
            den = (den + _ERFC_Q[i]) * y
        end
        r = (num + _ERFC_P[8]) / (den + _ERFC_Q[8])
        r *= exp(-y * y)
        return x < 0 ? 2.0 - r : r
    else
        z = 1.0 / (y * y)
        num = _ERFC_R[6] * z
        den = z
        @inbounds for i in 1:4
            num = (num + _ERFC_R[i]) * z
            den = (den + _ERFC_S[i]) * z
        end
        r = z * (num + _ERFC_R[5]) / (den + _ERFC_S[5])
        r = (0.5641895835477563 - r) / y      # 0.5641895835477563 = 1/√π
        r *= exp(-y * y)
        return x < 0 ? 2.0 - r : r
    end
end

# Combined Φ(x) and φ(x), sharing the single exp(-x²/2).  The mid/large erfc
# branches already form g = exp(-x²/2) internally; we reuse it for the pdf.
@inline function normcdf_pdf(x::Float64)
    xe = -x * INV_SQRT2        # erfc argument
    y = abs(xe)
    if y <= 0.46875
        z = y * y
        num = _ERF_P[5] * z
        den = z
        @inbounds for i in 1:3
            num = (num + _ERF_P[i]) * z
            den = (den + _ERF_Q[i]) * z
        end
        r = xe * (num + _ERF_P[4]) / (den + _ERF_Q[4])   # erf(xe)
        Φ = 0.5 * (1.0 - r)                               # 0.5*erfc(xe)
        g = exp(-0.5 * x * x)
        return Φ, INV_SQRT2PI * g
    elseif y <= 4.0
        num = _ERFC_P[9] * y
        den = y
        @inbounds for i in 1:7
            num = (num + _ERFC_P[i]) * y
            den = (den + _ERFC_Q[i]) * y
        end
        r = (num + _ERFC_P[8]) / (den + _ERFC_Q[8])
        g = exp(-y * y)                                   # = exp(-x²/2)
        ec = r * g
        erfc = xe < 0 ? 2.0 - ec : ec
        return 0.5 * erfc, INV_SQRT2PI * g
    else
        z = 1.0 / (y * y)
        num = _ERFC_R[6] * z
        den = z
        @inbounds for i in 1:4
            num = (num + _ERFC_R[i]) * z
            den = (den + _ERFC_S[i]) * z
        end
        r = z * (num + _ERFC_R[5]) / (den + _ERFC_S[5])
        r = (0.5641895835477563 - r) / y
        g = exp(-y * y)
        ec = r * g
        erfc = xe < 0 ? 2.0 - ec : ec
        return 0.5 * erfc, INV_SQRT2PI * g
    end
end

# Φ(x) when g = exp(-x²/2) is already known. The Cody rational factors
# erfc = R(y)·e^{-y²} on the mid/large ranges, so supplying g removes the `exp`
# there entirely; the small branch uses the erf rational and needs no exp anyway.
# CALLER MUST GUARANTEE  g == exp(-x²/2).
@inline function normcdf_withg(x::Float64, g::Float64)
    xe = -x * INV_SQRT2
    y = abs(xe)
    if y <= 0.46875
        z = y * y
        num = _ERF_P[5] * z
        den = z
        @inbounds for i in 1:3
            num = (num + _ERF_P[i]) * z
            den = (den + _ERF_Q[i]) * z
        end
        r = xe * (num + _ERF_P[4]) / (den + _ERF_Q[4])
        return 0.5 * (1.0 - r)
    elseif y <= 4.0
        num = _ERFC_P[9] * y
        den = y
        @inbounds for i in 1:7
            num = (num + _ERFC_P[i]) * y
            den = (den + _ERFC_Q[i]) * y
        end
        R = (num + _ERFC_P[8]) / (den + _ERFC_Q[8])
        ec = R * g
        return xe < 0 ? 0.5 * (2.0 - ec) : 0.5 * ec
    else
        z = 1.0 / (y * y)
        num = _ERFC_R[6] * z
        den = z
        @inbounds for i in 1:4
            num = (num + _ERFC_R[i]) * z
            den = (den + _ERFC_S[i]) * z
        end
        R = z * (num + _ERFC_R[5]) / (den + _ERFC_S[5])
        R = (0.5641895835477563 - R) / y
        ec = R * g
        return xe < 0 ? 0.5 * (2.0 - ec) : 0.5 * ec
    end
end

# Scaled complementary error function erfcx(x) = e^{x²} erfc(x), for x ≥ 0.
# Cheaper than erfc on the mid/large ranges: the Cody rational already factors
# erfc = R(y)·e^{-y²}, so erfcx = R(y) with no exp at all.
@inline function erfcx_pos(y::Float64)
    if y <= 0.46875
        z = y * y
        num = _ERF_P[5] * z
        den = z
        @inbounds for i in 1:3
            num = (num + _ERF_P[i]) * z
            den = (den + _ERF_Q[i]) * z
        end
        erf = y * (num + _ERF_P[4]) / (den + _ERF_Q[4])
        return exp(z) * (1.0 - erf)
    elseif y <= 4.0
        num = _ERFC_P[9] * y
        den = y
        @inbounds for i in 1:7
            num = (num + _ERFC_P[i]) * y
            den = (den + _ERFC_Q[i]) * y
        end
        return (num + _ERFC_P[8]) / (den + _ERFC_Q[8])
    else
        z = 1.0 / (y * y)
        num = _ERFC_R[6] * z
        den = z
        @inbounds for i in 1:4
            num = (num + _ERFC_R[i]) * z
            den = (den + _ERFC_S[i]) * z
        end
        r = z * (num + _ERFC_R[5]) / (den + _ERFC_S[5])
        return (0.5641895835477563 - r) / y
    end
end

# --- Inverse normal CDF: Acklam (2003), ~1.15e-9 relative accuracy -----------
# Plenty accurate for a *seed*; the polisher finishes the job.
const _AK_A = (-3.969683028665376e+01, 2.209460984245205e+02,
               -2.759285104469687e+02, 1.383577518672690e+02,
               -3.066479806614716e+01, 2.506628277459239e+00)
const _AK_B = (-5.447609879822406e+01, 1.615858368580409e+02,
               -1.556989798598866e+02, 6.680131188771972e+01,
               -1.328068155288572e+01)
const _AK_C = (-7.784894002430293e-03, -3.223964580411365e-01,
               -2.400758277161838e+00, -2.549732539343734e+00,
                4.374664141464968e+00,  2.938163982698783e+00)
const _AK_D = (7.784695709041462e-03, 3.224671290700398e-01,
               2.445134137142996e+00, 3.754408661907416e+00)

@inline function norminv(p::Float64)
    if p < 0.02425
        q = sqrt(-2.0 * log(p))
        return (((((_AK_C[1]*q+_AK_C[2])*q+_AK_C[3])*q+_AK_C[4])*q+_AK_C[5])*q+_AK_C[6]) /
               ((((_AK_D[1]*q+_AK_D[2])*q+_AK_D[3])*q+_AK_D[4])*q+1.0)
    elseif p <= 0.97575
        q = p - 0.5
        r = q * q
        return (((((_AK_A[1]*r+_AK_A[2])*r+_AK_A[3])*r+_AK_A[4])*r+_AK_A[5])*r+_AK_A[6])*q /
               (((((_AK_B[1]*r+_AK_B[2])*r+_AK_B[3])*r+_AK_B[4])*r+_AK_B[5])*r+1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((_AK_C[1]*q+_AK_C[2])*q+_AK_C[3])*q+_AK_C[4])*q+_AK_C[5])*q+_AK_C[6]) /
                ((((_AK_D[1]*q+_AK_D[2])*q+_AK_D[3])*q+_AK_D[4])*q+1.0)
    end
end
