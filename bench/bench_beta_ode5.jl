# Beta ODE5 central seed + y6 certificate: ns/q for the certified central region
# (headline), the mixed full band, and the raw zero-eval seed tier, against the
# existing certified/uncertified logit solvers and Distributions.
# Amortized per-(a,b) batches; preallocated, allocation-free; run with -O3.
include("../src/QuantileExpansions.jl"); using .QuantileExpansions
using BenchmarkTools, Printf
import Distributions

per(bmk, N) = round(minimum(bmk).time / N, digits=1)

function cover(a, b, us)
    c = 0
    for u in us
        c += beta_quantile_ode5_mode(a, b, u)
    end
    100 * c / length(us)
end

function main()
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.6
    # central band (mostly certified for qualifying shapes) and full band
    ctr = collect(range(0.15, 0.85, length=4096))
    full = collect(range(0.001, 0.999, length=4096))
    outc = similar(ctr); outf = similar(full)

    shapes = ((2.0,5.0), (4.0,4.0), (8.0,3.0), (20.0,12.5), (50.0,50.0), (100.0,100.0))

    tmp = similar(ctr)
    println("=== central band u in [0.15,0.85] (ns/q; precompute amortized once/shape) ===")
    @printf("%-11s %6s | %9s %9s %9s | %9s %9s | %8s\n",
            "(a,b)","cov%","ode5","cert(old)","full(old)","rawseed","Distr","pre(us)")
    for (a,b) in shapes
        cov = cover(a,b,ctr)
        o5!(o)   = beta_quantile_ode5_batch!(o,a,b,ctr)
        cert!(o) = beta_quantile_batch!(o,a,b,ctr; certified=true)
        full!(o) = beta_quantile_batch!(o,a,b,ctr; certified=false)
        seed!(o) = beta_ode5_seed_batch!(o,a,b,ctr)
        ds!(o)   = (D=Distributions.Beta(a,b); @inbounds for i in eachindex(ctr); o[i]=Distributions.quantile(D,ctr[i]); end; o)
        N = length(ctr)
        t_o5   = per(@benchmark($o5!($outc)), N)
        t_cert = per(@benchmark($cert!($outc)), N)
        t_full = per(@benchmark($full!($outc)), N)
        t_seed = per(@benchmark($seed!($outc)), N)
        t_ds   = per(@benchmark($ds!($outc)), N)
        t_pre  = round(minimum(@benchmark(BetaODE5($a,$b))).time/1000, digits=2)
        @printf("(%g,%g)%*s %6.1f | %9.1f %9.1f %9.1f | %9.1f %9.1f | %8.2f\n",
                a,b, max(1,5-length("($a,$b)")),"", cov, t_o5, t_cert, t_full, t_seed, t_ds, t_pre)
    end

    println("\n=== full band u in [0.001,0.999] (ns/q) ===")
    @printf("%-11s %6s | %9s %9s %9s | %9s\n",
            "(a,b)","cov%","ode5","cert(old)","full(old)","Distr")
    for (a,b) in shapes
        cov = cover(a,b,full)
        o5!(o)   = beta_quantile_ode5_batch!(o,a,b,full)
        cert!(o) = beta_quantile_batch!(o,a,b,full; certified=true)
        full!(o) = beta_quantile_batch!(o,a,b,full; certified=false)
        ds!(o)   = (D=Distributions.Beta(a,b); @inbounds for i in eachindex(full); o[i]=Distributions.quantile(D,full[i]); end; o)
        N = length(full)
        t_o5   = per(@benchmark($o5!($outf)), N)
        t_cert = per(@benchmark($cert!($outf)), N)
        t_full = per(@benchmark($full!($outf)), N)
        t_ds   = per(@benchmark($ds!($outf)), N)
        @printf("(%g,%g)%*s %6.1f | %9.1f %9.1f %9.1f | %9.1f\n",
                a,b, max(1,5-length("($a,$b)")),"", cov, t_o5, t_cert, t_full, t_ds)
    end
end
main()
