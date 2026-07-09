include("../src/core/specialfuns.jl")
include("../src/dists/blackscholes.jl")
using BenchmarkTools, Base.Threads
import SpecialFunctions: erfinv

function build_grid()
    vols = vcat(0.01, collect(0.05:0.05:2.0))
    deltas = [0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95]
    ks = Float64[]; cs = Float64[]
    for v in vols, D in deltas
        ppf = sqrt(2.0) * erfinv(2D - 1)
        k = v * (0.5 * v - ppf)
        d1 = -k / v + 0.5 * v; d2 = d1 - v
        c = normcdf(d1) - exp(k) * normcdf(d2)
        push!(ks, k); push!(cs, c)
    end
    return ks, cs
end

function bs_batch_serial!(out, ks, cs)
    @inbounds for i in eachindex(ks)
        out[i] = bs_implied_vol(ks[i], cs[i])
    end
    return out
end

function bs_batch_threaded!(out, ks, cs)
    @inbounds Threads.@threads :static for i in eachindex(ks)
        out[i] = bs_implied_vol(ks[i], cs[i])
    end
    return out
end

ks0, cs0 = build_grid()
reps = 5000
ks = repeat(ks0, reps); cs = repeat(cs0, reps)
out = similar(ks)
N = length(ks)
println("nthreads = ", nthreads(), "   N = ", N)

bs_batch_serial!(out, ks, cs)
bs_batch_threaded!(out, ks, cs)

bs = @benchmark bs_batch_serial!($out, $ks, $cs)
bt = @benchmark bs_batch_threaded!($out, $ks, $cs)
println("serial   per-IV: ", round(minimum(bs).time / N, digits=3), " ns")
println("threaded per-IV: ", round(minimum(bt).time / N, digits=3), " ns   (", round(minimum(bs).time/minimum(bt).time, digits=1), "x)")
