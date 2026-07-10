# x-space BetaQ vs logit-space BetaLogitQ vs Distributions: per-(a,b) batches,
# u-grid on [1e-8, 1-1e-8], accuracy = max |Δ logit x| vs Distributions
# (skipping points where x rounds to 0/1 in double).
include("../src/QuantileExpansions.jl"); using .QuantileExpansions
using BenchmarkTools, Printf
import Distributions

function main()
    us = collect(range(1e-8, 1.0-1e-8, length=4096))
    out = similar(us); N = length(us)
    t(b) = round(minimum(b).time/N, digits=1)
    @printf("%-12s %9s %9s %10s %13s %13s\n", "(a,b)", "x-sp ns", "logit ns", "Distr ns", "x-sp maxerr", "logit maxerr")
    for (a,b) in ((0.5,0.5), (0.75,2.0), (2.0,5.0), (5.0,0.2), (20.0,12.5), (100.0,100.0))
        xs!(o) = (@inbounds for i in 1:N; o[i]=beta_quantile(a,b,us[i]); end; o)
        lg!(o) = (@inbounds for i in 1:N; o[i]=beta_quantile_logit(a,b,us[i]); end; o)
        ds!(o) = (D=Distributions.Beta(a,b); @inbounds for i in 1:N; o[i]=Distributions.quantile(D,us[i]); end; o)
        xs!(out); lg!(out); ds!(out)
        D = Distributions.Beta(a,b)
        eo=0.0; el=0.0
        for i in 1:N
            xr = Distributions.quantile(D,us[i]); (xr<=0||xr>=1) && continue
            lr = log(xr)-log1p(-xr)
            xo = beta_quantile(a,b,us[i]); xl = beta_quantile_logit(a,b,us[i])
            (xo>0&&xo<1) && (eo=max(eo,abs(log(xo)-log1p(-xo)-lr)))
            (xl>0&&xl<1) && (el=max(el,abs(log(xl)-log1p(-xl)-lr)))
        end
        @printf("(%g,%g)%*s %9.1f %9.1f %10.1f %13.2e %13.2e\n", a, b,
                max(1, 10-length("($a,$b)")), "",
                t(@benchmark $xs!($out)), t(@benchmark $lg!($out)), t(@benchmark $ds!($out)), eo, el)
    end
end
main()
