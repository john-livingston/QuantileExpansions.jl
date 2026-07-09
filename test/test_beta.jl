include("../src/core/specialfuns.jl")
include("../src/core/solver.jl")
include("../src/dists/beta.jl")
using Statistics
import Distributions
import SpecialFunctions: beta_inc

function main()
    abs_ = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]
    ps = vcat(1e-6, 1e-4, 1e-3, 0.01, 0.05:0.05:0.95..., 0.99, 0.999, 1-1e-4, 1-1e-6)
    maxrel = 0.0; maxfwd = 0.0; aw=0.0; bw=0.0; pw=0.0; n=0; nbad=0
    function iters(D,p,tol)
        x=seed(D,p); k=0
        for _ in 1:12
            f,fp,φ2,ξ=hh_terms(D,x,p); k+=1; abs(f)<tol && break
            r=f/fp; den=-6+r*(6φ2-r*ξ); xn=abs(den)<1e-20 ? x-r : x+3r*(2-r*φ2)/den
            if !(isfinite(xn)&&0<xn<1); xn = x-r; if !(isfinite(xn)&&0<xn<1); xn=f>0 ? 0.5*x : 0.5*(x+1); end; end
            x=xn
        end; k
    end
    allits=Int[]
    for a in abs_, b in abs_
        D = Distributions.Beta(a, b)
        for p in ps
            xref = Distributions.quantile(D, p)
            x = beta_quantile(a, b, p)
            n += 1; push!(allits, iters(BetaQ(a,b),p,1e-13))
            rel = abs(x - xref) / max(abs(xref), 1e-300)
            if rel > maxrel; maxrel=rel; aw=a; bw=b; pw=p; end
            Ix, _ = beta_inc(a, b, x); maxfwd = max(maxfwd, abs(Ix - p))
            rel > 1e-9 && (nbad += 1)
        end
    end
    println("beta: n=", n, "  mean iters=", round(mean(allits),digits=2), "  max iters=", maximum(allits))
    println("  max rel err in x vs Distributions: ", maxrel, "  (a=",aw,",b=",bw,",p=",pw,")")
    println("  max forward |I_x(a,b)-p|: ", maxfwd)
    println("  nbad (rel>1e-9): ", nbad)
end
main()
