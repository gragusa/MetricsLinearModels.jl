##############################################################################
##
## Type OLSEstimator (for OLS estimation) - GLM.jl compatible structure
##
##############################################################################

"""
    OLSEstimator <: StatsAPI.RegressionModel

Model type for linear regression estimated by ordinary least squares (OLS).

Uses a GLM.jl-compatible structure with separate response (`rr`) and
predictor (`pp`) components, plus econometrics-specific fixed effects (`fes`).

Use `ols(df, formula)` to fit this model type.

# Fields
- `rr::OLSResponse{T}`: Response object (y, fitted values, weights)
- `pp::OLSLinearPredictor{T}`: Predictor object (X, coefficients, factorization)
- `fes::OLSFixedEffects{T}`: Fixed effects component
- `formula::FormulaTerm`: Original formula
- `formula_schema::FormulaTerm`: Schema for predict
- `contrasts::Dict`: Contrasts used
- `esample::BitVector`: Estimation sample indicator
- `coefnames::Vector{String}`: Coefficient names
- `basis_coef::BitVector`: Non-collinear coefficients indicator
- `nobs::Int`: Number of observations
- `dof::Int`: Degrees of freedom (parameters estimated)
- `dof_fes::Int`: DOF absorbed by fixed effects
- `dof_residual::Int`: Residual degrees of freedom
- `tss_total::T`: Total sum of squares (before FE)
- `tss_partial::T`: TSS after partialing out FEs
- `rss::T`: Residual sum of squares
- `r2::T`: R-squared
- `r2_within::T`: Within R-squared (with FEs)
- `has_intercept::Bool`: Whether model has intercept
- `F::T`: F-statistic
- `p::T`: P-value of F-statistic
"""
struct OLSEstimator{T <: AbstractFloat, P <: OLSLinearPredictor{T}} <:
       StatsAPI.RegressionModel
    # Core GLM-style components
    rr::OLSResponse{T}              # Response object
    pp::P                           # Predictor object (Chol or QR)

    # Fixed effects component
    fes::OLSFixedEffects{T}         # Fixed effects (empty if no FEs)

    # Formula and metadata
    formula::FormulaTerm
    formula_schema::FormulaTerm
    contrasts::Dict

    # Sample information
    esample::BitVector              # Which observations used

    # Coefficient metadata
    coefnames::Vector{String}
    basis_coef::BitVector           # Which coefficients are not collinear

    # Degrees of freedom
    nobs::Int                       # Number of observations
    dof::Int                        # Number of estimated parameters
    dof_fes::Int                    # DOF absorbed by fixed effects
    dof_residual::Int               # Residual degrees of freedom

    # Model fit statistics
    tss_total::T                    # Total sum of squares (before FE)
    tss_partial::T                  # TSS after partialing out FEs
    rss::T                          # Residual sum of squares
    r2::T                           # R-squared
    r2_within::T                    # Within R-squared (if FEs present)

    # Additional metadata
    has_intercept::Bool

    # Test statistics
    F::T                            # F-statistic
    p::T                            # P-value of F-stat
end

has_iv(::OLSEstimator) = false
has_fe(m::OLSEstimator) = has_fe(m.formula)
dof_fes(m::OLSEstimator) = m.dof_fes

"""
    has_matrices(m::OLSEstimator) -> Bool

Check if model has stored matrices (X, y, mu). Returns false if fit with save=:minimal.
"""
has_matrices(m::OLSEstimator) = m.pp.X !== nothing

# Property accessor for backward compatibility (iterations/converged stored in fes)
function Base.getproperty(m::OLSEstimator, s::Symbol)
    if s === :iterations
        return getfield(m, :fes).iterations
    elseif s === :converged
        return getfield(m, :fes).converged
    else
        return getfield(m, s)
    end
end

##############################################################################
##
## StatsAPI Interface - GLM-compatible
##
##############################################################################

# Basic accessors
function StatsAPI.coef(m::OLSEstimator)
    # Return 0.0 for collinear coefficients (backward compatibility)
    beta = copy(m.pp.beta)
    beta[.!m.basis_coef] .= zero(eltype(beta))
    return beta
end
StatsAPI.coefnames(m::OLSEstimator) = m.coefnames
StatsAPI.responsename(m::OLSEstimator) = m.rr.response_name

function StatsAPI.response(m::OLSEstimator)
    m.rr.y === nothing && error("Response vector not stored. Model was fit with save=:minimal.")
    return m.rr.y
end

function StatsAPI.fitted(m::OLSEstimator)
    m.rr.mu === nothing && error("Fitted values not stored. Model was fit with save=:minimal.")
    return m.rr.mu
end

function StatsAPI.modelmatrix(m::OLSEstimator)
    m.pp.X === nothing && error("Model matrix not stored. Model was fit with save=:minimal.")
    return m.pp.X
end

# Residuals (compute from y - mu, unweighted)
function StatsAPI.residuals(m::OLSEstimator)
    m.rr.y === nothing && error("Response vector not stored. Model was fit with save=:minimal. Cannot compute residuals.")
    m.rr.mu === nothing && error("Fitted values not stored. Model was fit with save=:minimal. Cannot compute residuals.")
    resid = m.rr.y .- m.rr.mu
    # Unweight residuals if model was estimated with weights
    if isweighted(m.rr)
        sqrtw = sqrt.(m.rr.wts)
        resid = resid ./ sqrtw
    end
    return resid
end

# Sample size and DOF
StatsAPI.nobs(m::OLSEstimator) = m.nobs
StatsAPI.dof(m::OLSEstimator) = m.dof
StatsAPI.dof_residual(m::OLSEstimator) = m.dof_residual

# Model fit
StatsAPI.deviance(m::OLSEstimator) = m.rss
StatsAPI.nulldeviance(m::OLSEstimator) = m.tss_total
StatsAPI.rss(m::OLSEstimator) = m.rss
StatsAPI.mss(m::OLSEstimator) = m.tss_total - m.rss
StatsAPI.r2(m::OLSEstimator) = m.r2
StatsAPI.islinear(m::OLSEstimator) = true

# Variance-covariance (default: HC1 for econometrics)
StatsAPI.vcov(m::OLSEstimator) = vcov(CovarianceMatrices.HC1(), m)

# Formula
StatsModels.formula(m::OLSEstimator) = m.formula_schema

##############################################################################
##
## Additional Methods
##
##############################################################################

function StatsAPI.loglikelihood(m::OLSEstimator)
    n = nobs(m)
    σ² = deviance(m) / n
    return -n/2 * (log(2π) + log(σ²) + 1)
end

function StatsAPI.nullloglikelihood(m::OLSEstimator)
    n = nobs(m)
    σ² = nulldeviance(m) / n
    return -n/2 * (log(2π) + log(σ²) + 1)
end

function nullloglikelihood_within(m::OLSEstimator)
    n = nobs(m)
    tss_within = deviance(m) / (1 - m.r2_within)
    return -n/2 * (log(2π * tss_within / n) + 1)
end

function StatsAPI.adjr2(model::OLSEstimator, variant::Symbol = :devianceratio)
    has_int = hasintercept(formula(model))
    k = dof(model) + dof_fes(model) + has_int
    if variant == :McFadden
        k = k - has_int - has_fe(model)
        ll = loglikelihood(model)
        ll0 = nullloglikelihood(model)
        1 - (ll - k)/ll0
    elseif variant == :devianceratio
        n = nobs(model)
        dev = deviance(model)
        dev0 = nulldeviance(model)
        1 - (dev*(n - (has_int | has_fe(model)))) / (dev0 * max(n - k, 1))
    else
        throw(ArgumentError("variant must be one of :McFadden or :devianceratio"))
    end
end

function StatsAPI.confint(se::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        m::OLSEstimator; level::Real = 0.95)
    scale = tdistinvcdf(StatsAPI.dof_residual(m), 1 - (1 - level) / 2)
    se = CovarianceMatrices.stderror(se, m)
    hcat(coef(m) - scale * se, coef(m) + scale * se)
end

##############################################################################
##
## Predict and Residuals (with new data)
##
##############################################################################

function StatsAPI.predict(m::OLSEstimator, data)
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))

    has_cont_fe_interaction(m.formula) &&
        throw(ArgumentError("Interaction of fixed effect and continuous variable detected in formula; this is currently not supported in `predict`"))

    cdata = StatsModels.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)

    # Use only non-collinear coefficients
    coef_valid = coef(m)[m.basis_coef]

    if all(nonmissings)
        out = Xnew[:, m.basis_coef] * coef_valid
    else
        out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(cdata)))
        out[nonmissings] = Xnew[:, m.basis_coef] * coef_valid
    end

    if has_fe(m)
        nrow(fe(m)) > 0 ||
            throw(ArgumentError("Model has no estimated fixed effects. To store estimates of fixed effects, run `ols` with the option save = :fe"))

        df = DataFrame(data; copycols = false)
        fes = leftjoin(select(df, m.fes.fe_names), dropmissing(unique(m.fes.fe));
            on = m.fes.fe_names, makeunique = true, matchmissing = :equal, order = :left)
        fes = combine(fes, AsTable(Not(m.fes.fe_names)) => sum)

        if any(ismissing, Matrix(select(df, m.fes.fe_names))) || any(ismissing, Matrix(fes))
            out = allowmissing(out)
        end

        out[nonmissings] .+= fes[nonmissings, 1]

        if any(.!nonmissings)
            out[.!nonmissings] .= missing
        end
    end

    return out
end

function StatsAPI.residuals(m::OLSEstimator, data)
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))
    has_fe(m) &&
        throw("To access residuals for a model with high-dimensional fixed effects, access them directly with `residuals(m)`.")

    cdata = StatsModels.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)
    y = response(m.formula_schema, cdata)

    # Use only non-collinear coefficients
    coef_valid = coef(m)[m.basis_coef]

    if all(nonmissings)
        out = y - Xnew[:, m.basis_coef] * coef_valid
    else
        out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(cdata)))
        out[nonmissings] = y - Xnew[:, m.basis_coef] * coef_valid
    end
    return out
end

"""
   fe(m::OLSEstimator; keepkeys = false)

Return a DataFrame with fixed effects estimates.
The output is aligned with the original DataFrame used in `ols`.

### Keyword arguments
* `keepkeys::Bool` : Should the returned DataFrame include the original variables used to define groups? Default to false
"""
function fe(m::OLSEstimator; keepkeys = false)
    !has_fe(m) && throw("fe() is not defined for models without fixed effects")
    if keepkeys
        m.fes.fe
    else
        m.fes.fe[!, (length(m.fes.fe_names) + 1):end]
    end
end

function StatsAPI.coeftable(m::OLSEstimator; level = 0.95)
    cc = coef(m)
    se = CovarianceMatrices.stderror(HC1(), m)
    coefnms = coefnames(m)
    conf_int = confint(HC1(), m; level = level)

    # put (intercept) last
    if !isempty(coefnms) &&
       ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end

    tt = cc ./ se
    CoefTable(
        hcat(cc, se, tt, fdistccdf.(Ref(1), Ref(StatsAPI.dof_residual(m)), abs2.(tt)),
            conf_int[:, 1:2]),
        ["Estimate", "Std. Error", "t-stat", "Pr(>|t|)", "Lower 95%", "Upper 95%"],
        ["$(coefnms[i])" for i in 1:length(cc)], 4)
end

##############################################################################
##
## Display Result
##
##############################################################################

function top(m::OLSEstimator)
    out = ["Number of obs" sprint(show, nobs(m), context = :compact => true);
           "Converged" m.fes.converged;
           "dof (model)" sprint(show, dof(m), context = :compact => true);
           "dof (residuals)" sprint(show, dof_residual(m), context = :compact => true);
           "R²" @sprintf("%.3f", r2(m));
           "R² adjusted" @sprintf("%.3f", adjr2(m));
           "F-statistic" sprint(show, m.F, context = :compact => true);
           "P-value" @sprintf("%.3f", m.p);]
    if has_fe(m)
        out = vcat(out,
            ["R² within" @sprintf("%.3f", m.r2_within);
             "Iterations" sprint(show, m.fes.iterations, context = :compact => true);])
    end
    return out
end

import StatsBase: NoQuote, PValue
function Base.show(io::IO, m::OLSEstimator)
    ct = coeftable(m)
    cols = ct.cols;
    rownms = ct.rownms;
    colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]", floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    mat = [j == 1 ? NoQuote(rownms[i]) :
           j-1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j - 1][i]))) :
           j-1 in ct.teststatcol ? TestStat(cols[j - 1][i]) :
           cols[j - 1][i] isa AbstractString ? NoQuote(cols[j - 1][i]) : cols[j - 1][i]
           for i in 1:nr, j in 1:(nc + 1)]
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
    interwidth = 2 + mod(totwidth, 2)
    for i in 1:(div(size(ctop, 1) - 1, 2) + 1)
        print(io, ctop[2 * i - 1, 1])
        print(io, lpad(ctop[2 * i - 1, 2], halfwidth - length(ctop[2 * i - 1, 1])))
        print(io, " " ^ interwidth)
        if size(ctop, 1) >= 2*i
            print(io, ctop[2 * i, 1])
            print(io, lpad(ctop[2 * i, 2], halfwidth - length(ctop[2 * i, 1])))
        end
        println(io)
    end

    println(io, repeat('=', totwidth))
    print(io, repeat(' ', sum(A[1])))
    for j in 1:length(colnms)
        print(io, "  ", lpad(colnms[j], sum(A[j + 1])))
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
function StatsModels.apply_schema(t::FormulaTerm, schema::StatsModels.Schema,
        Mod::Type{<:OLSEstimator}, has_fe_intercept)
    schema = StatsModels.FullRank(schema)
    if has_fe_intercept
        push!(schema.already, InterceptTerm{true}())
    end
    FormulaTerm(apply_schema(t.lhs, schema.schema, StatisticalModel),
        StatsModels.collect_matrix_terms(apply_schema(t.rhs, schema, StatisticalModel)))
end
