include("../src/core/specialfuns.jl"); include("../src/core/solver.jl"); include("../src/dists/inverse_gaussian.jl")
using BenchmarkTools, Statistics
import Distributions
function build()
    pars=[(1.0,0.5),(1.0,1.0),(1.0,3.0),(1.0,10.0),(1.0,50.0),(2.0,1.0),(0.5,2.0),(5.0,1.0),(3.0,0.3)]
    M=Float64[];L=Float64[];P=Float64[]
    for (μ,λ) in pars, p in 0.01:0.01:0.99
        push!(M,μ);push!(L,λ);push!(P,p)
    end; M,L,P
end
mine!(o,M,L,P)=(@inbounds for i in eachindex(M); o[i]=ig_quantile(M[i],L[i],P[i]); end; o)
lib!(o,M,L,P)=(@inbounds for i in eachindex(M); o[i]=Distributions.quantile(Distributions.InverseGaussian(M[i],L[i]),P[i]); end; o)
function main()
    M,L,P=build(); N=length(M); out=similar(M)
    mine!(out,M,L,P); lib!(out,M,L,P)
    bm=@benchmark mine!($out,$M,$L,$P); bl=@benchmark lib!($out,$M,$L,$P)
    println("IG points: ",N)
    println("mine          per-quantile: ",round(minimum(bm).time/N,digits=1)," ns")
    println("Distributions per-quantile: ",round(minimum(bl).time/N,digits=1)," ns")
    println("speedup: ",round(minimum(bl).time/minimum(bm).time,digits=2),"x")
end
main()
