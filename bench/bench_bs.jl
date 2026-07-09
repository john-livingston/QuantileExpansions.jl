include("../src/core/specialfuns.jl")
include("../src/dists/blackscholes.jl")
using BenchmarkTools
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

# batch kernel, allocation-free
function bs_batch!(out, ks, cs)
    @inbounds for i in eachindex(ks)
        out[i] = bs_implied_vol(ks[i], cs[i])
    end
    return out
end

ks0, cs0 = build_grid()
reps = 5000
ks = repeat(ks0, reps); cs = repeat(cs0, reps)
out = similar(ks)
N = length(ks)
println("N = ", N)

# warmup + correctness sanity
bs_batch!(out, ks, cs)

b = @benchmark bs_batch!($out, $ks, $cs)
tmin = minimum(b).time            # ns for whole batch
tmed = median(b).time
println("alloc: ", minimum(b).memory, " bytes")
println("per-IV (min):    ", round(tmin / N, digits=3), " ns")
println("per-IV (median): ", round(tmed / N, digits=3), " ns")
