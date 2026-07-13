# SIMD Temme gamma quantile batch vs the scalar log-space solver and Distributions.
# Fixed shape a (>= GAMMA_SIMD_AMIN), batch over u, the sampling workload. Reports
# ns per quantile for W in {2,4,8} x N in {2,3}. Local Apple Silicon is 2-wide
# NEON, so W>2 is emulated; run on x86 (AVX2/AVX-512) via CI for the wide datapoint.
include("../src/QuantileExpansions.jl"); using .QuantileExpansions
using BenchmarkTools, Printf
import Distributions

println("CPU: ", Sys.CPU_NAME, "  (", Sys.CPU_THREADS, " threads, ", Sys.ARCH, ")")

function main()
    us = collect(range(1e-6, 1.0 - 1e-6, length=16384))
    out = similar(us); N = length(us)
    t(b) = round(minimum(b).time / N, digits=1)
    @printf("%-6s %9s %9s | %s\n", "a", "scalar", "Distr", "SIMD Temme  ns/q  (N=2 | N=3),  W in {2,4,8}")
    for a in (20.0, 50.0, 100.0, 500.0)
        Dl = GammaLogQ(a); Gd = Distributions.Gamma(a, 1.0)
        scal!(o) = (@inbounds for i in 1:N; o[i] = exp(QuantileExpansions.solve(Dl, us[i]; tol=1e-14)); end; o)
        dist!(o) = (@inbounds for i in 1:N; o[i] = Distributions.quantile(Gd, us[i]); end; o)
        scal!(out); dist!(out)
        bs = t(@benchmark $scal!($out))
        bd = t(@benchmark $dist!($out))
        print(@sprintf("%-6g %9s %9s |", a, string(bs), string(bd)))
        for W in (2, 4, 8)
            b2 = t(@benchmark gamma_quantile_batch_simd!($out, $a, $us, Val(2), Val($W)))
            b3 = t(@benchmark gamma_quantile_batch_simd!($out, $a, $us, Val(3), Val($W)))
            print(@sprintf("  W%d: %s|%s", W, string(b2), string(b3)))
        end
        println()
    end
end
main()
