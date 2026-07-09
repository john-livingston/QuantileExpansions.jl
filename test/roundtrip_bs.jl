include("../src/core/specialfuns.jl")
include("../src/dists/blackscholes.jl")

# signed-k forward call price (reference for generating test data)
function bs_price_signed(k::Float64, v::Float64)
    d1 = -k / v + 0.5 * v
    d2 = d1 - v
    return normcdf(d1) - exp(k) * normcdf(d2)
end

function run(; ks, vs, label)
    maxerr = 0.0; kw = 0.0; vw = 0.0; cw = 0.0
    n = 0; nbad = 0
    sumiter = 0
    for k in ks, vt in vs
        c = bs_price_signed(k, vt)
        # skip degenerate prices (numerically 0 time value)
        ctv = c - max(1 - exp(k), 0.0)
        ctv <= 1e-300 && continue
        v = bs_implied_vol(k, c)
        err = abs(v - vt) / vt
        n += 1
        if err > maxerr; maxerr = err; kw = k; vw = vt; cw = c; end
        if err > 1e-6; nbad += 1; end
    end
    println(rpad(label, 24), "n=", n, "  max rel err=", maxerr,
            "  (k=", round(kw,digits=4), ", v=", round(vw,digits=4), ")  nbad(>1e-6)=", nbad)
end

run(ks = range(-2.0, 2.0, length=81),   vs = range(0.05, 1.5, length=101), label="moderate")
run(ks = range(-0.001, 0.001, length=21), vs = range(0.05, 1.5, length=101), label="ATM")
run(ks = range(-4.0, 4.0, length=161),   vs = range(0.02, 3.0, length=151), label="wide")
run(ks = range(-6.0, 6.0, length=121),   vs = range(0.01, 0.5, length=101), label="deep-OTM small-v")
