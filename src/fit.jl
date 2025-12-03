##############################################################################
##
## Main User-Facing API: ols() and iv()
##
##############################################################################

"""
    ols(df, formula; kwargs...) -> OLSEstimator

Estimate a linear model using Ordinary Least Squares (OLS).

Supports high-dimensional categorical variables (fixed effects) but not instrumental variables.
For IV models, use `iv(estimator, df, formula)`.

# Arguments
- `df`: a Table (e.g., DataFrame)
- `formula`: A formula created using `@formula(y ~ x1 + x2 + fe(group))`

# Keyword Arguments
- `contrasts::Dict = Dict()`: Contrast codings for categorical variables
- `weights::Union{Nothing, Symbol}`: Column name for weights
- `save::Symbol = :residuals`: Save residuals (`:residuals`), fixed effects (`:fe`), or both (`:all`)
- `save_cluster::Union{Symbol, Vector{Symbol}, Nothing}`: Cluster variables to save for post-estimation vcov
- `dof_add::Integer = 0`: Manual adjustment to degrees of freedom
- `method::Symbol = :cpu`: Computation method (`:cpu`, `:CUDA`, `:Metal`)
- `nthreads::Integer`: Number of threads (default: `Threads.nthreads()` for CPU)
- `double_precision::Bool = true` for CPU, `false` otherwise: Use Float64 vs Float32
- `tol::Real = 1e-6`: Tolerance for fixed effects demeaning
- `maxiter::Integer = 10000`: Maximum iterations for fixed effects
- `drop_singletons::Bool = true`: Drop singleton observations
- `progress_bar::Bool = true`: Show progress bar during estimation
- `subset::Union{Nothing, AbstractVector}`: Select specific rows

# Returns
- `OLSEstimator{T}`: Fitted model (T is Float64 or Float32 depending on `double_precision`)

# Examples
```julia
using DataFrames, RDatasets, MetricsLinearModels

df = dataset("plm", "Cigar")

# Simple OLS
model = ols(df, @formula(Sales ~ NDI + Pop))

# With fixed effects
model = ols(df, @formula(Sales ~ NDI + fe(State) + fe(Year)))

# Post-estimation robust standard errors
vcov(HC3(), model)
vcov(:State, :CR1, model)  # cluster-robust

# With weights
model = ols(df, @formula(Sales ~ NDI), weights = :Pop)
```

# Post-Estimation Variance Calculations

After fitting a model, you can compute different variance-covariance matrices without re-running the regression:

```julia
model = ols(df, @formula(y ~ x1 + x2 + fe(firm_id)))

# Different robust estimators
vcov(HC3(), model)           # Heteroskedasticity-robust (HC3)
vcov(HC1(), model)           # Default (HC1)

# Cluster-robust (using stored cluster variable from fe())
vcov(:firm_id, :CR1, model)

# Two-way clustering
vcov((:firm_id, :year), :CR1, model)

# HAC (time series)
vcov(Bartlett(5), model)

# Standard errors and coefficient table
stderror(HC3(), model)
coeftable(model, :firm_id, :CR1)
```

See also: [`iv`](@ref), [`OLSEstimator`](@ref)
"""
function ols(df, formula::FormulaTerm; kwargs...)
    has_iv(formula) &&
        throw(ArgumentError("Formula contains instrumental variables. Use `iv(TSLS(), df, formula)` instead."))
    return fit_ols(df, formula; kwargs...)
end

##############################################################################
##
## IV Function - Instrumental Variables
##
##############################################################################

"""
    iv(estimator::AbstractIVEstimator, df, formula; kwargs...) -> IVEstimator

Estimate an instrumental variables model using the specified estimator.

# Arguments
- `estimator::AbstractIVEstimator`: Estimator type (`TSLS()`, `LIML()`, etc.)
- `df`: a Table (e.g., DataFrame)
- `formula`: A formula with IV syntax: `@formula(y ~ x + (endo ~ instrument))`

# Keyword Arguments
Same as `ols()`, plus:
- `first_stage::Bool = true`: Compute first-stage F-statistics

# Returns
- `IVEstimator{T}`: Fitted IV model (T is Float64 or Float32 depending on `double_precision`)

# Available Estimators
- `TSLS()`: Two-Stage Least Squares (implemented)
- `LIML()`: Limited Information Maximum Likelihood (not yet implemented)

# Examples
```julia
using DataFrames, RDatasets, MetricsLinearModels

df = dataset("plm", "Cigar")

# Two-stage least squares
model = iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

# With fixed effects
model = iv(TSLS(), df, @formula(Sales ~ (Price ~ Pimin) + fe(State)))

# Post-estimation
vcov(HC3(), model)
coeftable(model, :State, :CR1)

# LIML (when implemented)
# model = iv(LIML(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))
```

# Post-Estimation Variance Calculations

IV models support the same post-estimation vcov calculations as OLS:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))

# Heteroskedasticity-robust
vcov(HC3(), model)

# Cluster-robust
vcov(:firm_id, :CR1, model)

# Two-way clustering
vcov((:firm_id, :year), :CR1, model)
```

See also: [`ols`](@ref), [`TSLS`](@ref), [`LIML`](@ref), [`IVEstimator`](@ref)
"""
function iv(estimator::AbstractIVEstimator, df, formula::FormulaTerm; kwargs...)
    !has_iv(formula) &&
        throw(ArgumentError("Formula does not contain instrumental variables. Use `ols(df, formula)` instead."))

    if estimator isa LIML
        fit_liml(df, formula; kwargs...)  # Will error with "not implemented"
    elseif estimator isa TSLS
        return fit_tsls(df, formula; kwargs...)
    else
        error("Unknown IV estimator: $(typeof(estimator))")
    end
end
