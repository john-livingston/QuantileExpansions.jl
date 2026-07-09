include("../src/core/specialfuns.jl")
include("../src/core/solver.jl")
include("../src/dists/gamma.jl")
using BenchmarkTools, Statistics
import Distributions

# realistic grid of (a, p)
function build()
    as = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    ps = collect(0.01:0.01:0.99)
    A = Float64[]; P = Float64[]
    for a in as, p in ps
        push!(A, a); push!(P, p)
    end
    A, P
end

# iteration count
function iters(D, p, tol)
    x = seed(D, p); n = 0
    for _ in 1:12
        f, fp, φ2, ξ = hh_terms(D, x, p); n += 1
        abs(f) < tol && break
        r = f/fp; denom = -6 + r*(6φ2 - r*ξ)
        xn = abs(denom)<1e-20 ? x-r : x + 3r*(2-r*φ2)/denom
        if !(isfinite(xn) && xn>1e-300); xn = x - r; end
        x = xn
    end
    n
end

mine!(o,A,P) = (@inbounds for i in eachindex(A); o[i] = gamma_quantile(A[i], P[i]); end; o)
function lib!(o,A,P)
    @inbounds for i in eachindex(A)
        o[i] = Distributions.quantile(Distributions.Gamma(A[i], 1.0), P[i])
    end
    o
end

function main()
    A, P = build(); N = length(A); out = similar(A)
    its = [iters(GammaQ(A[i]), P[i], 1e-13) for i in 1:N]
    println("gamma points: ", N, "  mean iters=", round(mean(its),digits=2), "  max=", maximum(its))
    mine!(out, A, P); lib!(out, A, P)
    bm = @benchmark mine!($out,$A,$P)
    bl = @benchmark lib!($out,$A,$P)
    println("mine          per-quantile: ", round(minimum(bm).time/N, digits=1), " ns")
    println("Distributions per-quantile: ", round(minimum(bl).time/N, digits=1), " ns")
    println("speedup: ", round(minimum(bl).time/minimum(bm).time, digits=2), "x")
end
main()
