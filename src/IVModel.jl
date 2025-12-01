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
## Post-Estimation vcov Methods
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

# Cluster-robust using symbol (looks up stored cluster variable)
vcov(:firm_id, :CR1, model)

# Two-way clustering
vcov((:firm_id, :year), :CR1, model)
```
"""
function StatsBase.vcov(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator{T}) where T
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored. Post-estimation vcov not available.")
    isnothing(m.residuals) && error("Model does not have residuals stored. Use save=:residuals or save=:all when fitting.")

    n = nobs(m)
    k = dof(m)
    invXX = m.postestimation.invXX
    resid = m.residuals

    # Homoskedastic variance
    if ve isa CovarianceMatrices.HR1 || ve isa CovarianceMatrices.HR0
        σ² = sum(abs2, resid) / dof_residual(m)
        return Symmetric(σ² * invXX)
    end

    # Heteroskedasticity-robust (HC0, HC1, etc.)
    X = m.postestimation.X
    meat = X' * Diagonal(resid.^2) * X

    # Degree of freedom adjustment for HC1
    scale = if ve isa CovarianceMatrices.HC1
        n / (n - k)
    elseif ve isa CovarianceMatrices.HC0
        one(T)
    else
        # Default to HC1 scaling for other types
        n / (n - k)
    end

    return Symmetric(scale * invXX * meat * invXX)
end

##############################################################################
##
## Convenience Methods for Cluster-Robust with Symbol Lookup
##
##############################################################################

"""
    vcov(cluster_var::Symbol, estimator_type::Symbol, m::IVEstimator)

Compute cluster-robust variance-covariance matrix using a stored cluster variable.
"""
function StatsBase.vcov(cluster_var::Symbol, estimator_type::Symbol, m::IVEstimator)
    # Look up cluster variable
    haskey(m.postestimation.cluster_vars, cluster_var) || _cluster_not_found_error(cluster_var, m)
    cluster_vec = m.postestimation.cluster_vars[cluster_var]

    if estimator_type == :CR0
        ve = CovarianceMatrices.CR0(cluster_vec)
    elseif estimator_type == :CR1
        ve = CovarianceMatrices.CR1(cluster_vec)
    elseif estimator_type == :CR2
        ve = CovarianceMatrices.CR2(cluster_vec)
    elseif estimator_type == :CR3
        ve = CovarianceMatrices.CR3(cluster_vec)
    else
        error("Unknown cluster-robust estimator type: $estimator_type. Use :CR0, :CR1, :CR2, or :CR3")
    end

    return vcov(ve, m)
end

"""
    vcov(cluster_vars::Tuple, estimator_type::Symbol, m::IVEstimator)

Compute multi-way cluster-robust variance-covariance matrix.
"""
function StatsBase.vcov(cluster_vars::Tuple, estimator_type::Symbol, m::IVEstimator)
    cluster_vecs = Tuple(begin
        haskey(m.postestimation.cluster_vars, var) || _cluster_not_found_error(var, m)
        m.postestimation.cluster_vars[var]
    end for var in cluster_vars)

    if estimator_type == :CR0
        ve = CovarianceMatrices.CR0(cluster_vecs)
    elseif estimator_type == :CR1
        ve = CovarianceMatrices.CR1(cluster_vecs)
    elseif estimator_type == :CR2
        ve = CovarianceMatrices.CR2(cluster_vecs)
    elseif estimator_type == :CR3
        ve = CovarianceMatrices.CR3(cluster_vecs)
    else
        error("Unknown cluster-robust estimator type: $estimator_type. Use :CR0, :CR1, :CR2, or :CR3")
    end

    return vcov(ve, m)
end

"""
    stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)

Compute standard errors using a specified variance estimator.
"""
function StatsBase.stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)
    return sqrt.(diag(vcov(ve, m)))
end

function StatsBase.stderror(cluster_var::Symbol, estimator_type::Symbol, m::IVEstimator)
    return sqrt.(diag(vcov(cluster_var, estimator_type, m)))
end

function StatsBase.stderror(cluster_vars::Tuple, estimator_type::Symbol, m::IVEstimator)
    return sqrt.(diag(vcov(cluster_vars, estimator_type, m)))
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

    To use a different cluster variable, either:
      1. Re-run regression with save_cluster=:$cluster_name
      2. Use manual subsetting: vcov(CR1(esample(model, df.$cluster_name)), model)
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
