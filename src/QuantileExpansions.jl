"""
    QuantileExpansions

A single, distribution-generic regime-split quantile solver. Every target is an
inverse-CDF problem F(x)=p solved by a closed-form regime seed + one universal
HH-4 (Householder order 3, quartic) polish. Each distribution supplies a tiny
interface (`seed`, `hh_terms`); the compiler monomorphizes `solve` into
specialized, allocation-free, C-speed code.

Targets: Black–Scholes implied vol, Inverse Gaussian, Gamma, Beta.
"""
module QuantileExpansions

include("core/specialfuns.jl")
include("core/solver.jl")
include("core/vecmath.jl")
include("dists/blackscholes.jl")
include("dists/blackscholes_simd.jl")
include("dists/inverse_gaussian.jl")
include("dists/inverse_gaussian_simd.jl")
include("dists/gamma.jl")
include("dists/gamma_log.jl")
include("dists/gamma_simd.jl")
include("dists/gamma_fast.jl")
include("dists/beta.jl")
include("dists/beta_logit.jl")
include("dists/beta_ode5.jl")

# generic interface
export QuantileProblem, solve, solve_certified, seed, hh_terms
# special functions
export normcdf, normpdf, norminv, normcdf_pdf, erfc_hi, erfcx_pos
# Black–Scholes
export BSCall, bs_implied_vol, bs_implied_vol_generic, bs_implied_vol_cert, bs_implied_vol_fixed, bs_implied_vol_grad, bs_price
export bs_implied_vol_fixed_batch!, bs_implied_vol_fixed_batch_threaded!, BSFixedWorkspace, vexp, vlog
# other distributions
export IGQ, ig_quantile, ig_quantile_cert, ig_quantile_batch!, ig_quantile_batch_threaded!, GammaQ, gamma_quantile, GammaLogQ, gamma_quantile_log, gamma_quantile_log_cert, gamma_quantile_batch!, GammaTemmeQ, gamma_quantile_batch_simd!, gamma_quantile_batch_simd_threaded!, GAMMA_SIMD_AMIN, GammaFast, gamma_quantile_fast, gamma_quantile_fast_batch!, BetaQ, beta_quantile, BetaLogitQ, beta_quantile_logit, beta_quantile_logit_cert, beta_quantile_batch!
# Beta ODE5 central seed + y6 certificate
export BetaODE5, beta_ode5_seed, beta_ode5_seed_batch!, beta_quantile_ode5, beta_quantile_ode5_batch!, beta_quantile_ode5_mode

end # module
