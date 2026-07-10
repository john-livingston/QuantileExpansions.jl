# Head-to-head: our x-space gamma solver (GammaQ, WH seed) vs the log-space
# port of Alper's engine (GammaLogQ: series/CF5/Mills seeds + density-scaled
# stop), vs Distributions.jl. Mirrors his benchmark protocol: fixed shape a,
# 16384-point u-grid on [1e-8, 1-1e-8], per-(a) batch amortization, single
# thread, ns per quantile, accuracy as max |Δ ln x| vs Distributions.
include("../src/QuantileExpansions.jl"); using .QuantileExpansions
using .QuantileExpansions: GammaLogQ, GammaQ, solve
using BenchmarkTools, Printf
import Distributions

function main()
    us = collect(range(1e-8, 1.0 - 1e-8, length=16384))
    out = similar(us); N = length(us)
    t(b) = round(minimum(b).time / N, digits=1)
    @printf("%-6s %10s %10s %12s %14s %14s\n", "a", "ours ns", "log ns", "Distr ns", "ours maxlogerr", "log maxlogerr")
    for a in (0.75, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0)
        Dx = GammaQ(a)                     # per-batch amortized, like his engine
        Dl = GammaLogQ(a)
        ours!(o) = (@inbounds for i in 1:N; o[i] = solve(Dx, us[i]; tol=1e-13); end; o)
        logs!(o) = a == 1.0 ?
            (@inbounds for i in 1:N; o[i] = -log1p(-us[i]); end; o) :
            (@inbounds for i in 1:N; o[i] = exp(solve(Dl, us[i]; tol=1e-14)); end; o)
        dist!(o) = (G = Distributions.Gamma(a,1.0); @inbounds for i in 1:N; o[i] = Distributions.quantile(G, us[i]); end; o)
        ours!(out); logs!(out); dist!(out)
        # accuracy vs Distributions (log-x metric, Alper's)
        G = Distributions.Gamma(a, 1.0)
        eo = 0.0; el = 0.0
        for i in 1:N
            xr = Distributions.quantile(G, us[i]); lr = log(xr)
            eo = max(eo, abs(log(solve(Dx, us[i]; tol=1e-13)) - lr))
            xl = a == 1.0 ? -log1p(-us[i]) : exp(solve(Dl, us[i]; tol=1e-14))
            el = max(el, abs(log(xl) - lr))
        end
        bo = @benchmark $ours!($out)
        bl = @benchmark $logs!($out)
        bd = @benchmark $dist!($out)
        @printf("%-6g %10.1f %10.1f %12.1f %14.2e %14.2e\n", a, t(bo), t(bl), t(bd), eo, el)
    end
end
main()
