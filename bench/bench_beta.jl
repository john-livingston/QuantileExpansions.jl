include("../src/core/specialfuns.jl")
include("../src/core/solver.jl")
include("../src/dists/beta.jl")
using BenchmarkTools, Statistics
import Distributions
import SpecialFunctions: beta_inc

function build()
    abs_ = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
    ps = collect(0.01:0.01:0.99)
    A = Float64[]; B = Float64[]; P = Float64[]
    for a in abs_, b in abs_, p in ps
        push!(A,a); push!(B,b); push!(P,p)
    end
    A, B, P
end

# count iterations through the actual (mirrored) solver path
function iters_mirror(a,b,p,tol)
    if p > 0.5; a,b,p = b,a,1-p; end
    D = BetaQ(a,b); x = seed(D,p); k = 0
    for _ in 1:12
        f,fp,φ2,ξ = hh_terms(D,x,p); k += 1; abs(f)<tol && break
        r=f/fp; den=-6+r*(6φ2-r*ξ); xn=abs(den)<1e-20 ? x-r : x+3r*(2-r*φ2)/den
        if !(isfinite(xn)&&0<xn<1); xn=x-r; if !(isfinite(xn)&&0<xn<1); xn=f>0 ? 0.5*x : 0.5*(x+1); end; end
        x=xn
    end
    k
end

mine!(o,A,B,P) = (@inbounds for i in eachindex(A); o[i]=beta_quantile(A[i],B[i],P[i]); end; o)
function lib!(o,A,B,P)
    @inbounds for i in eachindex(A); o[i]=Distributions.quantile(Distributions.Beta(A[i],B[i]),P[i]); end; o
end

function main()
    A,B,P = build(); N = length(A); out = similar(A)
    its = [iters_mirror(A[i],B[i],P[i],1e-13) for i in 1:N]
    # accuracy (skip unrepresentable boundary)
    mr = 0.0; nb = 0
    for i in 1:N
        xr = Distributions.quantile(Distributions.Beta(A[i],B[i]),P[i])
        (xr<=0 || xr>=1) && continue
        r = abs(beta_quantile(A[i],B[i],P[i])-xr)/xr
        mr = max(mr,r); r>1e-9 && (nb+=1)
    end
    println("beta points: ", N, "  mean iters=", round(mean(its),digits=2), "  max=", maximum(its))
    println("  max rel err vs Distributions (representable): ", mr, "  nbad>1e-9: ", nb)
    mine!(out,A,B,P); lib!(out,A,B,P)
    bm = @benchmark mine!($out,$A,$B,$P)
    bl = @benchmark lib!($out,$A,$B,$P)
    println("mine          per-quantile: ", round(minimum(bm).time/N,digits=1), " ns")
    println("Distributions per-quantile: ", round(minimum(bl).time/N,digits=1), " ns")
    println("speedup: ", round(minimum(bl).time/minimum(bm).time,digits=2), "x")
end
main()
