# SIMD batch kernel benchmark: scalar fixed-step vs vectorized batch at
# W ∈ {2,4,8}, plus the seed-pass cost alone (Amdahl accounting).
# Prints host CPU info first — run on x86 (AVX2/AVX-512) via CI for the
# wide-vector datapoint; local Apple Silicon is 2-wide NEON.
include("../src/QuantileExpansions.jl"); using .QuantileExpansions
using BenchmarkTools
import SpecialFunctions: erfinv

println("CPU: ", Sys.CPU_NAME, "  (", Sys.CPU_THREADS, " threads, ", Sys.ARCH, ")")
if Sys.islinux()
    flags = try
        m = match(r"flags\s*:\s*(.*)", read("/proc/cpuinfo", String))
        m === nothing ? "" : m.captures[1]
    catch; "" end
    simd_flags = filter(f -> occursin(r"^(sse|avx|fma)", f), split(flags))
    println("SIMD flags: ", join(simd_flags, " "))
end

function grid()
    vols = vcat(0.01, collect(0.05:0.05:2.0)); dl = [0.05,0.20,0.30,0.45,0.55,0.70,0.80,0.95]
    ks = Float64[]; cs = Float64[]
    for v in vols, D in dl
        k = v*(0.5v - sqrt(2)*erfinv(2D-1)); d1 = -k/v+0.5v; d2 = d1-v
        push!(ks, k); push!(cs, normcdf(d1) - exp(k)*normcdf(d2))
    end
    ks, cs
end

function main()
    k0, c0 = grid()
    ks = repeat(k0, 5000); cs = repeat(c0, 5000)
    out = similar(ks); N = length(ks); ws = BSFixedWorkspace(N)
    t(b) = round(minimum(b).time/N, digits=2)
    sc2!(o) = (@inbounds for i in 1:N; o[i] = bs_implied_vol_fixed(ks[i], cs[i], Val(2)); end; o)
    sc3!(o) = (@inbounds for i in 1:N; o[i] = bs_implied_vol_fixed(ks[i], cs[i], Val(3)); end; o)
    sc2!(out); sc3!(out)
    println("scalar fixed-2     : ", t(@benchmark $sc2!($out)), " ns/IV")
    println("scalar fixed-3     : ", t(@benchmark $sc3!($out)), " ns/IV")
    for W in (2, 4, 8)
        b2 = @benchmark bs_implied_vol_fixed_batch!($out, $ks, $cs, Val(2), Val($W); ws=$ws)
        b3 = @benchmark bs_implied_vol_fixed_batch!($out, $ks, $cs, Val(3), Val($W); ws=$ws)
        println("batch W=$W fixed-2  : ", t(b2), " ns/IV    fixed-3: ", t(b3), " ns/IV")
    end
    seed!(o) = (@inbounds for i in 1:N
        κ, cst, E, invE = QuantileExpansions._otm_reduce(ks[i], cs[i])
        ws.κ[i]=κ; ws.cstar[i]=cst; ws.E[i]=E; ws.invE[i]=invE
        o[i] = QuantileExpansions.bs_seed(κ, cst, E)
    end; o)
    seed!(out)
    println("seed pass alone    : ", t(@benchmark $seed!($out)), " ns/IV   (scalar; Amdahl floor for the batch)")
end
main()
