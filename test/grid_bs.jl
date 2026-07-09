include("../src/core/specialfuns.jl")
include("../src/dists/blackscholes.jl")
import SpecialFunctions: erfinv

# Reference delta-based grid (matches the paper's build_grid)
function build_grid()
    vols = vcat(0.01, collect(0.05:0.05:2.0))
    deltas = [0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95]
    ks = Float64[]; vs = Float64[]; cs = Float64[]
    for v in vols, D in deltas
        ppf = sqrt(2.0) * erfinv(2D - 1)          # Φ⁻¹(D), high accuracy
        k = v * (0.5 * v - ppf)
        d1 = -k / v + 0.5 * v; d2 = d1 - v
        c = normcdf(d1) - exp(k) * normcdf(d2)
        push!(ks, k); push!(vs, v); push!(cs, c)
    end
    return ks, vs, cs
end

function accuracy(ks, vs, cs)
    maxabs = 0.0; maxrel = 0.0; kw = 0.0; vw = 0.0
    for i in eachindex(ks)
        v = bs_implied_vol(ks[i], cs[i])
        ae = abs(v - vs[i]); re = ae / vs[i]
        if ae > maxabs; maxabs = ae; end
        if re > maxrel; maxrel = re; kw = ks[i]; vw = vs[i]; end
    end
    println("grid points: ", length(ks))
    println("max abs err in v: ", maxabs)
    println("max rel err in v: ", maxrel, "  at (k=", round(kw,digits=4), ", v=", round(vw,digits=4), ")")
end

ks, vs, cs = build_grid()
accuracy(ks, vs, cs)
