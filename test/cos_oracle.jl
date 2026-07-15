# Generic Fang-Oosterlee COS validation oracle.
#
# Any distribution with a known characteristic function phi(xi) = E[e^{i xi X}]
# and cumulants (k1..k4) can have its CDF and pdf cross-checked against this,
# independently of the implementation under test. It is a reusable validation tool
# for any target with a tractable characteristic function (a COS inversion of the
# CF; see the two-factor CIR-sum section of RESULTS.md for where it was first used).
#
# Usage:
#   pl = cos_oracle_plan(xi -> cf(xi), k1, k2, k3, k4; N=1024, L=10)
#   F  = oracle_cdf(pl, x)      # truncation range is [k1 - L*w, k1 + L*w],
#   f  = oracle_pdf(pl, x)      #   w = sqrt(k2 + sqrt(|k4|))
#
# Accuracy grows with N and with a range wide enough to hold the tails; it is
# spectral (machine precision at small N) for smooth densities and only algebraic
# when the density is non-smooth at a boundary (e.g. s^(a-1) at 0 for small shape).

module COSOracle

export cos_oracle_plan, oracle_cdf, oracle_pdf

struct COSOraclePlan
    a::Float64
    b::Float64
    A::Vector{Float64}          # density cosine coefficients
end

# cf: xi::Real -> Complex. cumulants k1..k4 set the Fang-Oosterlee range.
function cos_oracle_plan(cf, k1::Real, k2::Real, k3::Real, k4::Real;
                         N::Int = 1024, L::Real = 10.0)
    w = L * sqrt(k2 + sqrt(abs(k4)))
    a = float(k1 - w)
    b = float(k1 + w)
    invba = 1 / (b - a)
    A = Vector{Float64}(undef, N)
    @inbounds for k in 0:N-1
        u = k * pi * invba
        A[k+1] = 2 * invba * real(cf(u) * cis(-u * a))
    end
    return COSOraclePlan(a, b, A)
end

function oracle_cdf(pl::COSOraclePlan, x::Real)
    xf = float(x)
    xf <= pl.a && return 0.0
    ba = pl.b - pl.a
    d = xf - pl.a
    d > ba && (d = ba)
    acc = 0.5 * pl.A[1] * d                    # k=0 term (A0/2)(x-a)
    @inbounds for k in 1:length(pl.A)-1
        acc += pl.A[k+1] * ba / (k * pi) * sin(k * pi * d / ba)
    end
    return acc
end

function oracle_pdf(pl::COSOraclePlan, x::Real)
    xf = float(x)
    (xf <= pl.a || xf >= pl.b) && return 0.0
    ba = pl.b - pl.a
    d = xf - pl.a
    acc = 0.5 * pl.A[1]
    @inbounds for k in 1:length(pl.A)-1
        acc += pl.A[k+1] * cos(k * pi * d / ba)
    end
    return acc
end

end # module
