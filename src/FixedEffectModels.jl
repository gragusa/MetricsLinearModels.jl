
module FixedEffectModels


using DataFrames
using FixedEffects
using LinearAlgebra
using Printf
using Reexport
using PrecompileTools
using Statistics
using StatsAPI
using StatsBase
using StatsFuns
@reexport using StatsModels
using Tables
using Vcov  # Deprecated, will be removed in future version

# CovarianceMatrices.jl for post-estimation vcov
using CovarianceMatrices
using CovarianceMatrices: AbstractAsymptoticVarianceEstimator
using CovarianceMatrices: HC0, HC1, HC2, HC3, HC4, HC5
using CovarianceMatrices: CR0, CR1, CR2, CR3
using CovarianceMatrices: Bartlett, Parzen, QuadraticSpectral, TukeyHanning, Truncated
using CovarianceMatrices: Information, Misspecified
using CovarianceMatrices: Uncorrelated

include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")
include("utils/formula.jl")
include("FixedEffectModel.jl")
include("fit.jl")
include("partial_out.jl")

# Export from StatsBase
export coef, coefnames, coeftable, responsename, vcov, stderror, nobs, dof, dof_residual, r2,  r², adjr2, adjr², islinear, deviance, nulldeviance, rss, mss, confint, predict, residuals, fit,
    loglikelihood, nullloglikelihood, dof_fes


export reg,
partial_out,
fe,
FixedEffectModel,
has_iv,
has_fe,
Vcov,
esample  # Helper for subsetting vectors to estimation sample

# Re-export commonly used CovarianceMatrices.jl estimators
export HC0, HC1, HC2, HC3, HC4, HC5
export CR0, CR1, CR2, CR3
export Bartlett, Parzen, QuadraticSpectral, TukeyHanning, Truncated
export Uncorrelated

##############################################################################
##
## Helper Function for Subsetting to Estimation Sample
##
##############################################################################

"""
    esample(model::FixedEffectModel, v::AbstractVector)

Subset a vector `v` to the estimation sample used by `model`.
Useful for manually subsetting cluster variables for post-estimation vcov calculations.

# Arguments
- `model::FixedEffectModel`: A fitted model
- `v::AbstractVector`: A vector to subset (must have same length as original data)

# Returns
- A vector containing only elements corresponding to observations in the estimation sample

# Examples
```julia
# Fit model
model = reg(df, @formula(y ~ x1 + x2))

# Use a cluster variable that wasn't saved in the model
vcov(CR1(esample(model, df.firm_id)), model)

# Multi-way clustering with manual subsetting
vcov(CR1((esample(model, df.firm_id), esample(model, df.year))), model)
```

# Note
The `esample` field in FixedEffectModel is a BitVector indicating which rows
of the original dataframe were included in the estimation (after dropping
missing values, singletons, etc.).
"""
function esample(m::FixedEffectModel, v::AbstractVector)
    length(v) == length(m.esample) ||
        throw(ArgumentError("Vector length ($(length(v))) must match original data length ($(length(m.esample)))"))
    return v[m.esample]
end


@compile_workload begin
    df = DataFrame(x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], x2 = [1.0, 2.0, 4.0, 4.0, 3.0, 5.0], y = [3.0, 4.0, 4.0, 5.0, 1.0, 2.0], id = [1, 1, 2, 2, 3, 3])
    reg(df, @formula(y ~ x1 + x2))
    reg(df, @formula(y ~ x1 + fe(id)))
    # Post-estimation vcov with new API
    model = reg(df, @formula(y ~ x1))
    vcov(HC1(), model)
end




end
