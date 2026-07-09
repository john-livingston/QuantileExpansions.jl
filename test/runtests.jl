include("../src/QuantileExpansions.jl")
using .QuantileExpansions
using Test
import Distributions
import SpecialFunctions: erfinv, gamma_inc, beta_inc

@testset "special functions" begin
    import SpecialFunctions
    for x in range(-10, 10, length=2001)
        @test normcdf(x) ≈ 0.5*SpecialFunctions.erfc(-x/sqrt(2)) atol=1e-15 rtol=1e-13
    end
    for y in range(0, 50, length=2001)
        @test erfcx_pos(y) ≈ SpecialFunctions.erfcx(y) rtol=1e-13
    end
end

@testset "BS-IV round trip (paper grid)" begin
    vols = vcat(0.01, collect(0.05:0.05:2.0)); deltas = [0.05,0.20,0.30,0.45,0.55,0.70,0.80,0.95]
    maxerr = 0.0
    for v in vols, D in deltas
        k = v*(0.5v - sqrt(2)*erfinv(2D-1)); d1=-k/v+0.5v; d2=d1-v
        c = normcdf(d1) - exp(k)*normcdf(d2)
        maxerr = max(maxerr, abs(bs_implied_vol(k,c) - v))
        maxerr = max(maxerr, abs(bs_implied_vol_generic(k,c) - v))
    end
    @test maxerr < 1e-12
end

@testset "Gamma vs Distributions (forward residual)" begin
    mf = 0.0
    for a in [0.5,1.0,2.0,5.0,20.0,100.0], p in [1e-4,0.01,0.1,0.5,0.9,0.99,1-1e-4]
        x = gamma_quantile(a,p)
        mf = max(mf, abs(gamma_inc(a,x,0)[1] - p))
    end
    @test mf < 1e-12
end

@testset "Beta vs Distributions (forward residual)" begin
    mf = 0.0
    for a in [0.5,1.0,2.0,5.0,20.0], b in [0.5,1.0,2.0,5.0,20.0], p in [1e-3,0.01,0.1,0.5,0.9,0.99,0.999]
        x = beta_quantile(a,b,p)
        (0 < x < 1) && (mf = max(mf, abs(beta_inc(a,b,x)[1] - p)))
    end
    @test mf < 1e-12
end

@testset "Inverse Gaussian vs Distributions" begin
    mf = 0.0
    for (μ,λ) in [(1.0,0.5),(1.0,3.0),(1.0,50.0),(2.0,1.0),(3.0,0.3)], p in [1e-4,0.01,0.1,0.5,0.9,0.99,1-1e-4]
        x = ig_quantile(μ,λ,p)
        mf = max(mf, abs(Distributions.cdf(Distributions.InverseGaussian(μ,λ),x) - p))
    end
    @test mf < 1e-12
end

@testset "allocation-free scalar solvers" begin
    @test (@allocated bs_implied_vol(0.1, 0.06)) == 0
    @test (@allocated ig_quantile(1.0, 3.0, 0.7)) == 0
end
