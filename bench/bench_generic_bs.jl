include("../src/core/specialfuns.jl")
include("../src/core/solver.jl")
include("../src/dists/blackscholes.jl")
using BenchmarkTools
import SpecialFunctions: erfinv

function build_grid()
    vols = vcat(0.01, collect(0.05:0.05:2.0)); deltas = [0.05,0.20,0.30,0.45,0.55,0.70,0.80,0.95]
    ks = Float64[]; cs = Float64[]; vs = Float64[]
    for v in vols, D in deltas
        ppf = sqrt(2.0)*erfinv(2D-1); k = v*(0.5v-ppf); d1 = -k/v+0.5v; d2 = d1-v
        push!(ks,k); push!(cs, normcdf(d1)-exp(k)*normcdf(d2)); push!(vs,v)
    end
    ks, cs, vs
end

fgen!(o,k,c) = (@inbounds for i in eachindex(k); o[i] = bs_implied_vol_generic(k[i],c[i]); end; o)
fdir!(o,k,c) = (@inbounds for i in eachindex(k); o[i] = bs_implied_vol(k[i],c[i]); end; o)

function main()
    ks0, cs0, vs0 = build_grid()
    me = 0.0
    for i in eachindex(ks0); me = max(me, abs(bs_implied_vol_generic(ks0[i],cs0[i]) - vs0[i])); end
    println("generic BS max abs err: ", me)
    ks = repeat(ks0,5000); cs = repeat(cs0,5000); out = similar(ks); N = length(ks)
    fgen!(out,ks,cs); fdir!(out,ks,cs)
    bg = @benchmark fgen!($out,$ks,$cs)
    bd = @benchmark fdir!($out,$ks,$cs)
    println("generic per-IV: ", round(minimum(bg).time/N, digits=2), " ns")
    println("direct  per-IV: ", round(minimum(bd).time/N, digits=2), " ns")
end
main()
