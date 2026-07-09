include("../src/core/specialfuns.jl")
include("../src/core/solver.jl")
include("../src/dists/inverse_gaussian.jl")
using Statistics
import Distributions

function igcdf_ref(μ, λ, x)
    Distributions.cdf(Distributions.InverseGaussian(μ, λ), x)
end

function main()
    # (μ, λ) pairs spanning low→high shape ratio λ/μ
    pars = [(1.0,0.2),(1.0,0.5),(1.0,1.0),(1.0,3.0),(1.0,10.0),(1.0,50.0),
            (2.0,1.0),(0.5,2.0),(5.0,1.0),(1.0,100.0),(3.0,0.3)]
    ps = vcat(1e-6,1e-4,1e-3,0.01,0.05:0.05:0.95...,0.99,0.999,1-1e-4,1-1e-6)
    maxrel=0.0; maxfwd=0.0; mw=0.0; lw=0.0; pw=0.0; n=0; nbad=0
    allits=Int[]
    function iters(D,p,tol)
        x=seed(D,p); k=0
        for _ in 1:12
            f,fp,φ2,ξ=hh_terms(D,x,p); k+=1; abs(f)<tol && break
            r=f/fp; den=-6+r*(6φ2-r*ξ); xn=abs(den)<1e-20 ? x-r : x+3r*(2-r*φ2)/den
            if !(isfinite(xn)&&xn>1e-300); xn=x-r; if !(isfinite(xn)&&xn>1e-300); xn=f>0 ? 0.5*x : 2x; end; end
            x=xn
        end; k
    end
    for (μ,λ) in pars
        D = Distributions.InverseGaussian(μ,λ)
        for p in ps
            xref = Distributions.quantile(D,p)
            x = ig_quantile(μ,λ,p)
            n+=1; push!(allits, iters(IGQ(μ,λ),p,1e-13))
            rel = abs(x-xref)/max(abs(xref),1e-300)
            if rel>maxrel; maxrel=rel; mw=μ; lw=λ; pw=p; end
            maxfwd = max(maxfwd, abs(igcdf_ref(μ,λ,x)-p))
            rel>1e-9 && (nbad+=1)
        end
    end
    println("IG: n=",n,"  mean iters=",round(mean(allits),digits=2),"  max iters=",maximum(allits))
    println("  max rel err in x vs Distributions: ",maxrel,"  (μ=",mw,",λ=",lw,",p=",pw,")")
    println("  max forward |F(x)-p|: ",maxfwd)
    println("  nbad (rel>1e-9): ",nbad)
end
main()
