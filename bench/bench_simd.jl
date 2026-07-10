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
    out = similar(ks); N = length(ks)
    t(b) = round(minimum(b).time/N, digits=2)
    sc2!(o) = (@inbounds for i in 1:N; o[i] = bs_implied_vol_fixed(ks[i], cs[i], Val(2)); end; o)
    sc3!(o) = (@inbounds for i in 1:N; o[i] = bs_implied_vol_fixed(ks[i], cs[i], Val(3)); end; o)
    sc2!(out); sc3!(out)
    println("scalar fixed-2     : ", t(@benchmark $sc2!($out)), " ns/IV")
    println("scalar fixed-3     : ", t(@benchmark $sc3!($out)), " ns/IV")
    ws = BSFixedWorkspace(N)
    for W in (2, 4, 8)
        f2 = @benchmark bs_implied_vol_fixed_batch!($out, $ks, $cs, Val(2), Val($W); vector_seed=true)
        f3 = @benchmark bs_implied_vol_fixed_batch!($out, $ks, $cs, Val(3), Val($W); vector_seed=true)
        p2 = @benchmark bs_implied_vol_fixed_batch!($out, $ks, $cs, Val(2), Val($W); vector_seed=false, ws=$ws)
        p3 = @benchmark bs_implied_vol_fixed_batch!($out, $ks, $cs, Val(3), Val($W); vector_seed=false, ws=$ws)
        println("W=$W fused    fixed-2: ", t(f2), " ns/IV   fixed-3: ", t(f3), " ns/IV")
        println("W=$W two-pass fixed-2: ", t(p2), " ns/IV   fixed-3: ", t(p3), " ns/IV")
    end
    b0 = @benchmark bs_implied_vol_fixed_batch!($out, $ks, $cs, Val(0), Val(8); vector_seed=true)
    println("vector seed alone  : ", t(b0), " ns/IV   (fused reduce+seed, W=8, zero updates)")
end
main()
