include("../src/core/specialfuns.jl")
include("../src/core/solver.jl")
include("../src/dists/gamma.jl")
import Distributions
import SpecialFunctions: gamma_inc

function main()
    as = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0, 30.0, 100.0, 500.0]
    ps = vcat(1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05:0.05:0.95..., 0.99, 0.999, 1-1e-4, 1-1e-6, 1-1e-8)
    maxrel = 0.0; maxfwd = 0.0; aw = 0.0; pw = 0.0; nbad = 0; n = 0
    for a in as
        D = Distributions.Gamma(a, 1.0)
        for p in ps
            xref = Distributions.quantile(D, p)
            x = gamma_quantile(a, p)
            n += 1
            rel = abs(x - xref) / max(abs(xref), 1e-300)
            if rel > maxrel; maxrel = rel; aw = a; pw = p; end
            # forward residual
            Pp, _ = gamma_inc(a, x, 0)
            maxfwd = max(maxfwd, abs(Pp - p))
            rel > 1e-9 && (nbad += 1)
        end
    end
    println("gamma: n=", n)
    println("  max rel err in x   vs Distributions: ", maxrel, "  (a=", aw, ", p=", pw, ")")
    println("  max forward |P(a,x)-p|: ", maxfwd)
    println("  nbad (rel>1e-9): ", nbad)
end
main()
