# Consolidated accuracy + speed report for all four distributions.
# Run:  julia --project=. -O3 -t <N> bench/run_all.jl
include("../src/QuantileExpansions.jl")
using .QuantileExpansions
using BenchmarkTools, Statistics, Base.Threads
import Distributions
import SpecialFunctions: erfinv, gamma_inc, beta_inc

perIV(b, N) = round(minimum(b).time / N, digits=1)

# ---------------- Black–Scholes ----------------
function bs_grid()
    vols = vcat(0.01, collect(0.05:0.05:2.0)); deltas = [0.05,0.20,0.30,0.45,0.55,0.70,0.80,0.95]
    ks=Float64[]; cs=Float64[]; vs=Float64[]
    for v in vols, D in deltas
        ppf=sqrt(2.0)*erfinv(2D-1); k=v*(0.5v-ppf); d1=-k/v+0.5v; d2=d1-v
        push!(ks,k); push!(cs, normcdf(d1)-exp(k)*normcdf(d2)); push!(vs,v)
    end; ks,cs,vs
end
bs_ser!(o,k,c)=(@inbounds for i in eachindex(k); o[i]=bs_implied_vol_generic(k[i],c[i]); end; o)
bs_par!(o,k,c)=(@inbounds Threads.@threads :static for i in eachindex(k); o[i]=bs_implied_vol_generic(k[i],c[i]); end; o)
bs_f2!(o,k,c) =(@inbounds for i in eachindex(k); o[i]=bs_implied_vol_fixed(k[i],c[i],Val(2)); end; o)
bs_f3!(o,k,c) =(@inbounds for i in eachindex(k); o[i]=bs_implied_vol_fixed(k[i],c[i],Val(3)); end; o)
bs_f2p!(o,k,c)=(@inbounds Threads.@threads :static for i in eachindex(k); o[i]=bs_implied_vol_fixed(k[i],c[i],Val(2)); end; o)

function report_bs()
    k0,c0,v0 = bs_grid()
    err(f) = maximum(abs(f(k0[i],c0[i])-v0[i]) for i in eachindex(k0))
    me = err(bs_implied_vol_generic)
    e2 = err((k,c)->bs_implied_vol_fixed(k,c,Val(2)))
    e3 = err((k,c)->bs_implied_vol_fixed(k,c,Val(3)))
    ks=repeat(k0,5000); cs=repeat(c0,5000); out=similar(ks); N=length(ks)
    for f in (bs_ser!,bs_par!,bs_f2!,bs_f3!,bs_f2p!); f(out,ks,cs); end
    println("BS-IV adaptive   max|Δv|=", me, "   serial=", perIV(@benchmark(bs_ser!($out,$ks,$cs)),N),
            " ns   threaded(", nthreads(), ")=", perIV(@benchmark(bs_par!($out,$ks,$cs)),N), " ns")
    println("BS-IV fixed-2    max|Δv|=", e2, "   serial=", perIV(@benchmark(bs_f2!($out,$ks,$cs)),N),
            " ns   threaded(", nthreads(), ")=", perIV(@benchmark(bs_f2p!($out,$ks,$cs)),N), " ns   [branch-free, fast mode]")
    println("BS-IV fixed-3    max|Δv|=", e3, "   serial=", perIV(@benchmark(bs_f3!($out,$ks,$cs)),N),
            " ns   [branch-free, full precision]")
end

# ---------------- helpers for the library-compared dists ----------------
function report_dist(name, mine, lib, args, fwd)
    N=length(args[1]); out=zeros(N)
    minef!(o)=(@inbounds for i in 1:N; o[i]=mine(map(a->a[i],args)...); end; o)
    libf!(o)=(@inbounds for i in 1:N; o[i]=lib(map(a->a[i],args)...); end; o)
    minef!(out); libf!(out)
    # accuracy: forward residual + rel-vs-library on representable points
    mr=0.0; mf=0.0
    for i in 1:N
        x=mine(map(a->a[i],args)...); xr=lib(map(a->a[i],args)...)
        mf=max(mf, fwd(map(a->a[i],args)..., x))
        (0<xr<Inf) && (mr=max(mr, abs(x-xr)/max(abs(xr),1e-300)))
    end
    bm=@benchmark $minef!($out); bl=@benchmark $libf!($out)
    println(rpad(name,11), " max|F(x)-p|=", mf, "   mine=", perIV(bm,N), " ns   lib=", perIV(bl,N),
            " ns   speedup=", round(minimum(bl).time/minimum(bm).time,digits=2), "x")
end

function main()
    println("="^96)
    println("Regime-split quantile solver — all distributions   (threads=", nthreads(), ")")
    println("="^96)
    report_bs()

    # Gamma
    A=Float64[];P=Float64[]
    for a in [0.5,0.8,1.0,1.5,2.0,3.0,5.0,10.0,20.0,50.0,100.0], p in 0.01:0.01:0.99; push!(A,a);push!(P,p); end
    report_dist("Gamma", (a,p)->gamma_quantile(a,p), (a,p)->Distributions.quantile(Distributions.Gamma(a,1.0),p),
                (A,P), (a,p,x)->abs(gamma_inc(a,x,0)[1]-p))

    # Beta
    Ab=Float64[];Bb=Float64[];Pb=Float64[]
    for a in [0.5,0.8,1.0,1.5,2.0,3.0,5.0,10.0,20.0], b in [0.5,0.8,1.0,1.5,2.0,3.0,5.0,10.0,20.0], p in 0.01:0.02:0.99
        push!(Ab,a);push!(Bb,b);push!(Pb,p)
    end
    report_dist("Beta", (a,b,p)->beta_quantile(a,b,p), (a,b,p)->Distributions.quantile(Distributions.Beta(a,b),p),
                (Ab,Bb,Pb), (a,b,p,x)->abs(beta_inc(a,b,x)[1]-p))

    # Inverse Gaussian
    M=Float64[];L=Float64[];Pi=Float64[]
    for (μ,λ) in [(1.0,0.5),(1.0,1.0),(1.0,3.0),(1.0,10.0),(1.0,50.0),(2.0,1.0),(0.5,2.0),(5.0,1.0),(3.0,0.3)], p in 0.01:0.01:0.99
        push!(M,μ);push!(L,λ);push!(Pi,p)
    end
    report_dist("InvGauss", (μ,λ,p)->ig_quantile(μ,λ,p),
                (μ,λ,p)->Distributions.quantile(Distributions.InverseGaussian(μ,λ),p),
                (M,L,Pi), (μ,λ,p,x)->abs(Distributions.cdf(Distributions.InverseGaussian(μ,λ),x)-p))
    println("="^96)
end
main()
