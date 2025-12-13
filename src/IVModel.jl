##############################################################################
##
## Type IVEstimator (for IV estimation)
##
##############################################################################

"""
    PostEstimationDataIV{T}

Container for post-estimation data required for IV variance-covariance calculations.

# Fields
- `X::Matrix{T}`: Design matrix used for inference (with predicted endogenous)
- `Xhat::Matrix{T}`: Original matrix with actual endogenous variables
- `crossx::Cholesky{T, Matrix{T}}`: Cholesky factorization of X'X
- `invXX::Symmetric{T, Matrix{T}}`: Inverse of X'X
- `weights::AbstractWeights`: Weights used in estimation
- `cluster_vars::NamedTuple`: Cluster variables (subsetted to esample)
- `basis_coef::BitVector`: Indicator of which coefficients are not collinear
"""
struct PostEstimationDataIV{T, W<:AbstractWeights}
    X::Matrix{T}
    Xhat::Matrix{T}
    crossx::Cholesky{T, Matrix{T}}
    invXX::Symmetric{T, Matrix{T}}
    weights::W
    cluster_vars::NamedTuple
    basis_coef::BitVector
end

"""
    IVEstimator <: StatsAPI.RegressionModel

Model type for instrumental variables regression.

Use `iv(estimator, df, formula)` to fit this model type, where `estimator` is
one of `TSLS()`, `LIML()`, etc.

# Examples
```julia
iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))
```
"""
struct IVEstimator{T} <: StatsAPI.RegressionModel
    estimator::AbstractIVEstimator  # Which IV estimator was used

    coef::Vector{T}   # Vector of coefficients

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    residuals::Union{AbstractVector, Nothing}
    fe::DataFrame

    # Post-estimation data for CovarianceMatrices.jl
    postestimation::Union{PostEstimationDataIV{T}, Nothing}

    fekeys::Vector{Symbol}

    coefnames::Vector       # Name of coefficients
    responsename::Union{String, Symbol} # Name of dependent variable
    formula::FormulaTerm        # Original formula
    formula_schema::FormulaTerm # Schema for predict
    contrasts::Dict

    nobs::Int64             # Number of observations
    dof::Int64              # Number parameters estimated - has_intercept. Used for p-value of F-stat.
    dof_fes::Int64          # Number of fixed effects
    dof_residual::Int64     # dof used for t-test and p-value of F-stat. nobs - degrees of freedoms with simple std
    rss::T            # Sum of squared residuals
    tss::T            # Total sum of squares

    # for FE
    iterations::Int         # Number of iterations
    converged::Bool         # Has the demeaning algorithm converged?
    r2_within::T      # within r2 (with fixed effect)

    # Test statistics
    F::T                    # F-statistic
    p::T                    # P-value of F-stat
    F_kp::T                 # Kleibergen-Paap first-stage F-stat
    p_kp::T                 # P-value of first-stage F-stat
end

has_iv(::IVEstimator) = true
has_fe(m::IVEstimator) = has_fe(m.formula)

##############################################################################
##
## StatsAPI Interface
##
##############################################################################

function StatsAPI.coef(m::IVEstimator)
    # Return 0.0 for collinear coefficients (backward compatibility)
    if !isnothing(m.postestimation) && !isempty(m.postestimation.basis_coef)
        beta = copy(m.coef)
        beta[.!m.postestimation.basis_coef] .= zero(eltype(beta))
        return beta
    else
        return m.coef
    end
end
StatsAPI.coefnames(m::IVEstimator) = m.coefnames
StatsAPI.responsename(m::IVEstimator) = m.responsename
StatsAPI.vcov(m::IVEstimator) = vcov(CovarianceMatrices.HC1(), m)  # Default to HC1
StatsAPI.nobs(m::IVEstimator) = m.nobs
StatsAPI.dof(m::IVEstimator) = m.dof
StatsAPI.dof_residual(m::IVEstimator) = m.dof_residual
StatsAPI.r2(m::IVEstimator) = r2(m, :devianceratio)
StatsAPI.islinear(m::IVEstimator) = true
StatsAPI.deviance(m::IVEstimator) = rss(m)
StatsAPI.nulldeviance(m::IVEstimator) = m.tss
StatsAPI.rss(m::IVEstimator) = m.rss
StatsAPI.mss(m::IVEstimator) = nulldeviance(m) - rss(m)
StatsModels.formula(m::IVEstimator) = m.formula_schema
dof_fes(m::IVEstimator) = m.dof_fes

##############################################################################
##
## CovarianceMatrices.jl Interface for Post-Estimation vcov
##
##############################################################################

"""
    CovarianceMatrices.momentmatrix(m::IVEstimator)

Returns the moment matrix for the model (X .* residuals).
Required for post-estimation variance-covariance calculations.
"""
function CovarianceMatrices.momentmatrix(m::IVEstimator)
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored. Post-estimation vcov not available.")
    isnothing(m.residuals) && error("Model does not have residuals stored. Use save=:residuals or save=:all when fitting.")
    return m.postestimation.X .* m.residuals
end

"""
    CovarianceMatrices.score(m::IVEstimator)

Returns the score matrix (Jacobian of moment conditions) for IV: -X'X/n.
"""
# function CovarianceMatrices.hessian_objective(m::IVEstimator)
#     isnothing(m.X) && error("Model does not have design matrix stored. Post-estimation vcov not available.")
#     return -Symmetric(m.X' * m.X) / m.nobs
# end

"""
    CovarianceMatrices.objective_hessian(m::IVEstimator)

Returns the Hessian of the least squares objective function: X'X/n.
"""
# function CovarianceMatrices.hessian_objective(m::IVEstimator)
#     isnothing(m.X) && error("Model does not have design matrix stored. Post-estimation vcov not available.")
#     return Symmetric(m.X' * m.X) / m.nobs
# end

##############################################################################
##
## CovarianceMatrices.jl aVar Interface for IVEstimator
##
##############################################################################

# Local alias for CovarianceMatrices (also defined in covariance.jl for OLS)
const _CM = CovarianceMatrices

"""
    bread(m::IVEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator for IV.
Uses the predicted endogenous variables (Xhat) in the design matrix.
"""
bread(m::IVEstimator) = m.postestimation.invXX

"""
    leverage(m::IVEstimator)

Compute leverage values (diagonal of hat matrix) for IV models.
For IV: h_i = X_i * (X'X)^(-1) * X_i' where X contains predicted endogenous.
"""
function leverage(m::IVEstimator)
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored.")
    X = m.postestimation.X
    invXX = m.postestimation.invXX
    # h_i = X_i * invXX * X_i'
    return vec(sum(X .* (X * invXX), dims=2))
end

# Residual adjustments for HC/HR estimators
# Note: HC0 = HR0 and HC1 = HR1 in CovarianceMatrices.jl (type aliases)
@noinline residualadjustment(k::_CM.HR0, m::IVEstimator) = 1.0  # Also handles HC0
@noinline residualadjustment(k::_CM.HR1, m::IVEstimator) = sqrt(nobs(m) / dof_residual(m))  # Also handles HC1
@noinline residualadjustment(k::_CM.HR2, m::IVEstimator) = 1.0 ./ sqrt.(1 .- leverage(m))  # Also handles HC2
@noinline residualadjustment(k::_CM.HR3, m::IVEstimator) = 1.0 ./ (1 .- leverage(m))  # Also handles HC3

@noinline function residualadjustment(k::_CM.HC4, m::IVEstimator)
    n = nobs(m)
    h = leverage(m)
    p = round(Int, sum(h))
    adj = similar(h)
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        adj[j] = 1 / (1 - h[j])^(delta / 2)
    end
    adj
end

@noinline function residualadjustment(k::_CM.HC5, m::IVEstimator)
    n = nobs(m)
    h = leverage(m)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    adj = similar(h)
    @inbounds for j in eachindex(h)
        alpha = min(n * h[j] / p, mx)
        adj[j] = 1 / (1 - h[j])^(alpha / 4)
    end
    adj
end

# Cluster-robust residual adjustments
@noinline residualadjustment(k::_CM.CR0, m::IVEstimator) = 1.0
@noinline residualadjustment(k::_CM.CR1, m::IVEstimator) = 1.0

# CR2 and CR3 for IV - leverage-adjusted cluster-robust
function residualadjustment(k::_CM.CR2, m::IVEstimator)
    @assert length(k.g) == 1 "CR2 for IV currently only supports single-way clustering"
    g = k.g[1]
    X = m.postestimation.X
    resid = m.residuals
    u = copy(resid)
    XX = bread(m)
    for groups in 1:g.ngroups
        ind = findall(x -> x == groups, g)
        Xg = view(X, ind, :)
        ug = view(u, ind)
        Hgg = Xg * XX * Xg'
        # Apply (I - H_gg)^(-1/2) to residuals
        F = cholesky!(Symmetric(I - Hgg); check=false)
        if issuccess(F)
            ldiv!(ug, F.L, ug)
        end
    end
    return u ./ resid
end

function residualadjustment(k::_CM.CR3, m::IVEstimator)
    @assert length(k.g) == 1 "CR3 for IV currently only supports single-way clustering"
    g = k.g[1]
    X = m.postestimation.X
    resid = m.residuals
    u = copy(resid)
    XX = bread(m)
    for groups in 1:g.ngroups
        ind = findall(g .== groups)
        Xg = view(X, ind, :)
        ug = view(u, ind)
        Hgg = Xg * XX * Xg'
        # Apply (I - H_gg)^(-1) to residuals
        F = cholesky!(Symmetric(I - Hgg); check=false)
        if issuccess(F)
            ldiv!(ug, F, ug)
        end
    end
    return u ./ resid
end

"""
    CovarianceMatrices.aVar(k, m::IVEstimator)

Compute the asymptotic variance matrix for IV estimation.
"""
function _CM.aVar(
        k::K,
        m::IVEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: _CM.AbstractAsymptoticVarianceEstimator}
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored.")
    isnothing(m.residuals) && error("Model does not have residuals stored.")

    # Compute adjusted moment matrix
    u = residualadjustment(k, m)
    M = m.postestimation.X .* m.residuals
    if !(u isa Number && u == 1.0)
        M = M .* u
    end

    # Compute aVar using CovarianceMatrices
    Σ = _CM.aVar(k, M; demean=demean, prewhite=prewhite, scale=scale)
    return Σ
end

# Disambiguating method for cluster-robust estimators
function _CM.aVar(
        k::K,
        m::IVEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: _CM.CR}
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored.")
    isnothing(m.residuals) && error("Model does not have residuals stored.")

    # Compute adjusted moment matrix
    u = residualadjustment(k, m)
    M = m.postestimation.X .* m.residuals
    if !(u isa Number && u == 1.0)
        M = M .* u
    end

    # Compute aVar using CovarianceMatrices
    Σ = _CM.aVar(k, M; demean=demean, prewhite=prewhite, scale=scale)
    return Σ
end

##############################################################################
##
## Post-Estimation vcov Methods
##
## Primary API (CovarianceMatrices.jl standard):
##   vcov(CR1(cluster_vec), model)
##   stderror(CR1(cluster_vec), model)
##
## Convenience API (for stored cluster variables):
##   vcov(:ClusterVar, :CR1, model)   # looks up cluster from model
##   stderror(:ClusterVar, :CR1, model)
##
##############################################################################

"""
    vcov(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)

Compute variance-covariance matrix using a specified estimator from CovarianceMatrices.jl.

# Supported Estimators
- **Heteroskedasticity-robust**: `HC0`, `HC1`, `HC2`, `HC3`, `HC4`, `HC5`
- **Cluster-robust**: `CR0`, `CR1`, `CR2`, `CR3`
- **HAC**: `Bartlett(bw)`, `Parzen(bw)`, `QuadraticSpectral(bw)`, etc.

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))

# Heteroskedasticity-robust
vcov(HC3(), model)

# Cluster-robust (standard CovarianceMatrices.jl API)
vcov(CR1(df.firm_id[model.esample]), model)

# Two-way clustering
vcov(CR1((df.firm_id[model.esample], df.year[model.esample])), model)

# Convenience API (for stored cluster variables - avoids manual subsetting)
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster=:firm_id)
vcov(:firm_id, :CR1, model)
```
"""
function StatsBase.vcov(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator{T}) where T
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored. Post-estimation vcov not available.")
    isnothing(m.residuals) && error("Model does not have residuals stored. Use save=:residuals or save=:all when fitting.")

    n = nobs(m)
    k = dof(m)
    B = bread(m)
    resid = m.residuals

    # Homoskedastic variance
    if ve isa CovarianceMatrices.HR0
        σ² = sum(abs2, resid) / n
        return Symmetric(σ² * B)
    elseif ve isa CovarianceMatrices.HR1
        σ² = sum(abs2, resid) / dof_residual(m)
        return Symmetric(σ² * B)
    end

    # Sandwich variance: V = B * A * B where A = aVar(k, m)
    A = _CM.aVar(ve, m)

    # Scale factor depends on estimator type
    # Note: HC1 = HR1 in CovarianceMatrices.jl
    scale = if ve isa _CM.HR1
        # HC1/HR1: n/(n-k) adjustment
        p_total = dof(m) + dof_fes(m)
        n * dof_residual(m) / (n - p_total)
    elseif ve isa Union{_CM.CR0, _CM.CR1, _CM.CR2, _CM.CR3}
        # Cluster-robust: use fixest-style correction
        _cluster_robust_scale_iv(ve, m, n)
    else
        # HC0/HR0, HC2/HR2, HC3/HR3, HC4, HC5: no additional scale (adjustment in residualadjustment)
        convert(T, n)
    end

    Σ = scale .* B * A * B
    return Symmetric(Σ)
end

"""
    _cluster_robust_scale_iv(k::_CM.CR, m::IVEstimator, n::Int)

Compute the scale factor for cluster-robust variance estimation for IV models.
Uses fixest-style small sample correction.
"""
function _cluster_robust_scale_iv(k::_CM.CR, m::IVEstimator, n::Int)
    cluster_groups = k.g
    G = minimum(g.ngroups for g in cluster_groups)

    # G/(G-1) adjustment - only for CR1, CR2, CR3
    G_adj = k isa _CM.CR0 ? 1.0 : G / (G - 1)

    # For IV, K = k (number of params) - we don't have FE nesting logic for IV
    # This is a simpler case since IV models typically don't have absorbed FE DOF
    K = dof(m)
    K_adj = (n - 1) / (n - K)

    return convert(Float64, n * G_adj * K_adj)
end

##############################################################################
##
## Symbol-Based Cluster-Robust Variance API
##
## When CR types are constructed with Symbol(s) instead of data vectors,
## these methods look up the cluster data from the model's stored clusters.
##
## Usage:
##   vcov(CR1(:StateID), model)           # single cluster
##   vcov(CR1(:StateID, :YearID), model)  # multi-way clustering
##
##############################################################################

"""
    _lookup_cluster_vecs_iv(cluster_syms::Tuple{Vararg{Symbol}}, m::IVEstimator)

Look up cluster vectors from stored cluster data in the IV model.
Returns a tuple of vectors corresponding to the requested cluster symbols.
"""
function _lookup_cluster_vecs_iv(cluster_syms::Tuple{Vararg{Symbol}}, m::IVEstimator)
    return Tuple(begin
        haskey(m.postestimation.cluster_vars, name) || _cluster_not_found_error(name, m)
        m.postestimation.cluster_vars[name]
    end for name in cluster_syms)
end

"""
    vcov(v::CovarianceMatrices.CR0{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR0 estimator using stored cluster variable(s).

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster = :firm_id)
vcov(CR0(:firm_id), model)
vcov(CR0(:firm_id, :year), model)  # multi-way
```
"""
function CovarianceMatrices.vcov(v::CovarianceMatrices.CR0{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR0(cluster_vecs), m)
end

"""
    vcov(v::CovarianceMatrices.CR1{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR1 estimator using stored cluster variable(s).

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster = :firm_id)
vcov(CR1(:firm_id), model)
vcov(CR1(:firm_id, :year), model)  # multi-way
```
"""
function CovarianceMatrices.vcov(v::CovarianceMatrices.CR1{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR1(cluster_vecs), m)
end

"""
    vcov(v::CovarianceMatrices.CR2{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR2 (leverage-adjusted) estimator using stored cluster variable(s).
"""
function CovarianceMatrices.vcov(v::CovarianceMatrices.CR2{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR2(cluster_vecs), m)
end

"""
    vcov(v::CovarianceMatrices.CR3{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR3 (squared leverage) estimator using stored cluster variable(s).
"""
function CovarianceMatrices.vcov(v::CovarianceMatrices.CR3{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR3(cluster_vecs), m)
end

"""
    stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)

Compute standard errors using a specified variance estimator.
"""
function StatsBase.stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)
    return sqrt.(diag(vcov(ve, m)))
end

##############################################################################
##
## Helper Functions
##
##############################################################################

function _cluster_not_found_error(cluster_name::Symbol, m::IVEstimator)
    available = isempty(m.postestimation.cluster_vars) ? "none" : join(keys(m.postestimation.cluster_vars), ", :")
    error("""
    Cluster variable :$cluster_name not found in model.

    Available cluster variables: :$available

    To use this cluster variable, either:
      1. Re-fit with save_cluster=:$cluster_name
      2. Use data directly: vcov(CR1(df.$cluster_name[model.esample]), model)
    """)
end

##############################################################################
##
## Additional Methods
##
##############################################################################

function StatsAPI.loglikelihood(m::IVEstimator)
    n = nobs(m)
    -n/2 * (log(2π * deviance(m) / n) + 1)
end

function StatsAPI.nullloglikelihood(m::IVEstimator)
    n = nobs(m)
    -n/2 * (log(2π * nulldeviance(m) / n) + 1)
end

function nullloglikelihood_within(m::IVEstimator)
    n = nobs(m)
    tss_within = deviance(m) / (1 - m.r2_within)
    -n/2 * (log(2π * tss_within / n) + 1)
end

function StatsAPI.adjr2(model::IVEstimator, variant::Symbol=:devianceratio)
    has_int = hasintercept(formula(model))
    k = dof(model) + dof_fes(model) + has_int
    if variant == :McFadden
        k = k - has_int - has_fe(model)
        ll = loglikelihood(model)
        ll0 = nullloglikelihood(model)
        1 - (ll - k)/ll0
    elseif variant == :devianceratio
        n = nobs(model)
        dev  = deviance(model)
        dev0 = nulldeviance(model)
        1 - (dev*(n - (has_int | has_fe(model)))) / (dev0 * max(n - k, 1))
    else
        throw(ArgumentError("variant must be one of :McFadden or :devianceratio"))
    end
end

function StatsAPI.confint(m::IVEstimator; level::Real = 0.95)
    scale = tdistinvcdf(StatsAPI.dof_residual(m), 1 - (1 - level) / 2)
    se = stderror(m)
    hcat(m.coef -  scale * se, m.coef + scale * se)
end

##############################################################################
##
## Predict and Residuals
##
##############################################################################

# Note: is_cont_fe_int() and has_cont_fe_interaction() are defined in utils/fit_common.jl

function StatsAPI.predict(m::IVEstimator, data)
    Tables.istable(data) ||
          throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))

    has_cont_fe_interaction(m.formula) &&
        throw(ArgumentError("Interaction of fixed effect and continuous variable detected in formula; this is currently not supported in `predict`"))

    cdata = StatsModels.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)
    if all(nonmissings)
        out = Xnew * m.coef
    else
        out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(cdata)))
        out[nonmissings] = Xnew * m.coef
    end

    if has_fe(m)
        nrow(fe(m)) > 0 || throw(ArgumentError("Model has no estimated fixed effects. To store estimates of fixed effects, run `iv` with the option save = :fe"))

        df = DataFrame(data; copycols = false)
        fes = leftjoin(select(df, m.fekeys), dropmissing(unique(m.fe)); on = m.fekeys,
                            makeunique = true, matchmissing = :equal, order = :left)
        fes = combine(fes, AsTable(Not(m.fekeys)) => sum)

        if any(ismissing, Matrix(select(df, m.fekeys))) || any(ismissing, Matrix(fes))
            out = allowmissing(out)
        end

        out[nonmissings] .+= fes[nonmissings, 1]

        if any(.!nonmissings)
            out[.!nonmissings] .= missing
        end
    end

    return out
end

function StatsAPI.residuals(m::IVEstimator, data)
    Tables.istable(data) ||
      throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))
    has_fe(m) &&
     throw("To access residuals for a model with high-dimensional fixed effects,  run `m = iv(..., save = :residuals)` and then access residuals with `residuals(m)`.")
    cdata = StatsModels.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)
    y = response(m.formula_schema, cdata)
    if all(nonmissings)
        out =  y -  Xnew * m.coef
    else
        out = Vector{Union{Float64, Missing}}(missing,  length(Tables.rows(cdata)))
        out[nonmissings] = y -  Xnew * m.coef
    end
    return out
end

function StatsAPI.residuals(m::IVEstimator)
    if m.residuals === nothing
        has_fe(m) && throw("To access residuals in a fixed effect regression,  run `iv` with the option save = :residuals, and then access residuals with `residuals()`")
        !has_fe(m) && throw("To access residuals,  use residuals(m, data) where `m` is an estimated IVEstimator and  `data` is a Table")
    end
    m.residuals
end

"""
   fe(m::IVEstimator; keepkeys = false)

Return a DataFrame with fixed effects estimates.
"""
function fe(m::IVEstimator; keepkeys = false)
   !has_fe(m) && throw("fe() is not defined for models without fixed effects")
   if keepkeys
       m.fe
   else
      m.fe[!, (length(m.fekeys)+1):end]
   end
end

function StatsAPI.coeftable(m::IVEstimator; level = 0.95)
    cc = coef(m)
    se = stderror(m)
    coefnms = coefnames(m)
    conf_int = confint(m; level = level)
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    CoefTable(
        hcat(cc, se, tt, fdistccdf.(Ref(1), Ref(StatsAPI.dof_residual(m)), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std. Error","t-stat", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4)
end

##############################################################################
##
## Display Result
##
##############################################################################

function top(m::IVEstimator)
    out = [
            "Number of obs" sprint(show, nobs(m), context = :compact => true);
            "Converged" m.converged;
            "dof (model)" sprint(show, dof(m), context = :compact => true);
            "dof (residuals)" sprint(show, dof_residual(m), context = :compact => true);
            "R²" @sprintf("%.3f",r2(m));
            "R² adjusted" @sprintf("%.3f",adjr2(m));
            "F-statistic" sprint(show, m.F, context = :compact => true);
            "P-value" @sprintf("%.3f",m.p);
            ]
    # Always show first-stage diagnostics for IV models
    out = vcat(out,
        [
            "F-statistic (first stage)" sprint(show, m.F_kp, context = :compact => true);
            "P-value (first stage)" @sprintf("%.3f",m.p_kp);
        ])
    if has_fe(m)
        out = vcat(out,
            [
                "R² within" @sprintf("%.3f",m.r2_within);
                "Iterations" sprint(show, m.iterations, context = :compact => true);
             ])
    end
    return out
end

import StatsBase: NoQuote, PValue
function Base.show(io::IO, m::IVEstimator)
    ct = coeftable(m)
    cols = ct.cols; rownms = ct.rownms; colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    mat = [j == 1 ? NoQuote(rownms[i]) :
           j-1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j-1][i]))) :
           j-1 in ct.teststatcol ? TestStat(cols[j-1][i]) :
           cols[j-1][i] isa AbstractString ? NoQuote(cols[j-1][i]) : cols[j-1][i]
           for i in 1:nr, j in 1:nc+1]
    io = IOContext(io, :compact=>true, :limit=>false)
    A = Base.alignment(io, mat, 1:size(mat, 1), 1:size(mat, 2),
                       typemax(Int), typemax(Int), 3)
    nmswidths = pushfirst!(length.(colnms), 0)
    A = [nmswidths[i] > sum(A[i]) ? (A[i][1]+nmswidths[i]-sum(A[i]), A[i][2]) : A[i]
         for i in 1:length(A)]
    totwidth = sum(sum.(A)) + 2 * (length(A) - 1)

    ctitle = string(typeof(m))
    halfwidth = div(totwidth - length(ctitle), 2)
    print(io, " " ^ halfwidth * ctitle * " " ^ halfwidth)
    ctop = top(m)
    for i in 1:size(ctop, 1)
        ctop[i, 1] = ctop[i, 1] * ":"
    end
    println(io, '\n', repeat('=', totwidth))
    halfwidth = div(totwidth, 2) - 1
    interwidth = 2 +  mod(totwidth, 2)
    for i in 1:(div(size(ctop, 1) - 1, 2)+1)
        print(io, ctop[2*i-1, 1])
        print(io, lpad(ctop[2*i-1, 2], halfwidth - length(ctop[2*i-1, 1])))
        print(io, " " ^interwidth)
        if size(ctop, 1) >= 2*i
            print(io, ctop[2*i, 1])
            print(io, lpad(ctop[2*i, 2], halfwidth - length(ctop[2*i, 1])))
        end
        println(io)
    end

    println(io, repeat('=', totwidth))
    print(io, repeat(' ', sum(A[1])))
    for j in 1:length(colnms)
        print(io, "  ", lpad(colnms[j], sum(A[j+1])))
    end
    println(io, '\n', repeat('─', totwidth))
    for i in 1:size(mat, 1)
        Base.print_matrix_row(io, mat, A, i, 1:size(mat, 2), "  ")
        i != size(mat, 1) && println(io)
    end
    println(io, '\n', repeat('=', totwidth))
    nothing
end

##############################################################################
##
## Schema
##
##############################################################################
function StatsModels.apply_schema(t::FormulaTerm, schema::StatsModels.Schema, Mod::Type{IVEstimator}, has_fe_intercept)
    schema = StatsModels.FullRank(schema)
    if has_fe_intercept
        push!(schema.already, InterceptTerm{true}())
    end
    FormulaTerm(apply_schema(t.lhs, schema.schema, StatisticalModel),
                StatsModels.collect_matrix_terms(apply_schema(t.rhs, schema, StatisticalModel)))
end
