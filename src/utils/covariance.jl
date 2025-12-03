const CM = CovarianceMatrices

##############################################################################
##
## CovarianceMatrices.jl Interface for Post-Estimation vcov
##
##############################################################################

"""
    CovarianceMatrices.momentmatrix(m::OLSEstimator)

Returns the moment matrix for the model (X .* residuals).
Required for post-estimation variance-covariance calculations.
"""
function CovarianceMatrices.momentmatrix(m::OLSEstimator)
    # Get residuals and model matrix from new structure
    resid = residuals(m)
    X = modelmatrix(m)
    return X .* resid
end

function CM.aVar(
        k::K,
        m::OLSEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: CM.AbstractAsymptoticVarianceEstimator}
    CM.setkernelweights!(k, m)
    mm = begin
        u = residualadjustment(k, m)
        M = copy(momentmatrix(m))
        @. M = M * u
        M
    end
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ

    Σ_out = similar(Σ)
    fill!(Σ_out, NaN)
    for j in axes(Σ, 1)
        for i in axes(Σ, 2)
            if basis_coef[j] && basis_coef[i]
                Σ_out[j, i] = Σ[j, i]
            end
        end
    end
    return Σ_out
end

# Disambiguating method for cluster-robust estimators (CR <: AbstractAsymptoticVarianceEstimator)
# This resolves ambiguity between:
#   - aVar(k::K, m::OLSEstimator) where K <: AbstractAsymptoticVarianceEstimator (above)
#   - aVar(k::CR, m::RegressionModel) from CovarianceMatrices.jl
function CM.aVar(
        k::K,
        m::OLSEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: CM.CR}
    mm = begin
        u = residualadjustment(k, m)
        M = copy(momentmatrix(m))
        @. M = M * u
        M
    end
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ

    Σ_out = similar(Σ)
    fill!(Σ_out, NaN)
    for j in axes(Σ, 1)
        for i in axes(Σ, 2)
            if basis_coef[j] && basis_coef[i]
                Σ_out[j, i] = Σ[j, i]
            end
        end
    end
    return Σ_out
end

function CM.setkernelweights!(
        k::CM.HAC{T},
        X::OLSEstimator
) where {T <: Union{CM.NeweyWest, CM.Andrews}}
    CM.setkernelweights!(k, modelmatrix(X))
    k.wlock .= true
end

##############################################################################
##
## Convenience Methods for Cluster-Robust with Symbol Lookup
##
##############################################################################

"""
    vcov(cluster_var::Symbol, estimator_type::Symbol, m::OLSEstimator)

Compute cluster-robust variance-covariance matrix using a stored cluster variable.

# Arguments
- `cluster_var::Symbol`: Name of the cluster variable (must be stored in model)
- `estimator_type::Symbol`: Type of cluster-robust estimator (`:CR0`, `:CR1`, `:CR2`, or `:CR3`)
- `m::OLSEstimator`: Fitted model
"""
function StatsBase.vcov(cluster_var::Symbol, estimator_type::Symbol, m::OLSEstimator)
    # Look up cluster variable from fes component
    haskey(m.fes.clusters, cluster_var) || _cluster_not_found_error(cluster_var, m)
    cluster_vec = m.fes.clusters[cluster_var]

    # Create appropriate estimator
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
    vcov(cluster_vars::Tuple, estimator_type::Symbol, m::OLSEstimator)

Compute multi-way cluster-robust variance-covariance matrix.
"""
function StatsBase.vcov(cluster_vars::Tuple, estimator_type::Symbol, m::OLSEstimator)
    # Look up all cluster variables from fes component
    cluster_vecs = Tuple(begin
                             haskey(m.fes.clusters, var) ||
                                 _cluster_not_found_error(var, m)
                             m.fes.clusters[var]
                         end
    for var in cluster_vars)

    # Create appropriate estimator
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
    stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::OLSEstimator)

Compute standard errors using a specified variance estimator.
"""
function StatsBase.stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::OLSEstimator)
    return sqrt.(diag(vcov(ve, m)))
end

# Convenience method for cluster-robust with symbol
function StatsBase.stderror(cluster_var::Symbol, estimator_type::Symbol, m::OLSEstimator)
    return sqrt.(diag(vcov(cluster_var, estimator_type, m)))
end

# Convenience method for multi-way clustering
function StatsBase.stderror(cluster_vars::Tuple, estimator_type::Symbol, m::OLSEstimator)
    return sqrt.(diag(vcov(cluster_vars, estimator_type, m)))
end

##############################################################################
##
## Helper Functions for Cluster Variable Handling
##
##############################################################################

# Error message for missing cluster variable
function _cluster_not_found_error(cluster_name::Symbol, m::OLSEstimator)
    available = isempty(m.fes.clusters) ? "none" : join(keys(m.fes.clusters), ", :")
    error("""
    Cluster variable :$cluster_name not found in model.

    Available cluster variables: :$available

    To use a different cluster variable, either:
      1. Re-run regression with save_cluster=:$cluster_name
      2. Use manual subsetting: vcov(CR1(esample(model, df.$cluster_name)), model)
    """)
end

"""
    bread(m::OLSEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator.
"""
bread(m::OLSEstimator) = invchol(m.pp)

"""
    leverage(m::OLSEstimator)

Compute leverage values (diagonal of hat matrix H = X(X'X)^(-1)X').
Uses h_i = ||X_i * U^(-1)||^2 where X'X = U'U for efficiency.
"""
function leverage(m::OLSEstimator{T, <:OLSPredictorChol}) where {T}
    # For Cholesky: X'X = U'U, so (X'X)^(-1) = U^(-1) * U^(-T)
    # h_i = X_i * U^(-1) * U^(-T) * X_i' = ||X_i * U^(-1)||^2
    X = modelmatrix(m)
    return vec(sum(abs2, X / m.pp.chol.U, dims = 2))
end

function leverage(m::OLSEstimator{T, <:OLSPredictorQR}) where {T}
    # For QR: X = QR, so X(X'X)^(-1)X' = QQ'
    # h_i = ||Q_i||^2
    X = modelmatrix(m)
    Q = Matrix(m.pp.qr.Q)[:, 1:size(m.pp.qr.R, 1)]
    return vec(sum(abs2, Q, dims = 2))
end

@noinline residualadjustment(k::CM.HR0, r::OLSEstimator) = 1.0
@noinline residualadjustment(k::CM.HR1, r::OLSEstimator) = √nobs(r) / √dof_residual(r)
@noinline residualadjustment(k::CM.HR2, r::OLSEstimator) = 1.0 ./ (1 .- leverage(r)) .^ 0.5
@noinline residualadjustment(k::CM.HR3, r::OLSEstimator) = 1.0 ./ (1 .- leverage(r))

@noinline function residualadjustment(k::CM.HR4, r::OLSEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::CM.HR4m, r::OLSEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(1, n * h[j] / p) + min(1.5, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::CM.HR5, r::OLSEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n * h[j] / p, mx)
        h[j] = 1 / (1 - h[j])^(alpha / 4)
    end
    return h
end

# For cluster-robust estimators CR0/CR1, no adjustment to moment matrix needed.
# The clustering is handled by CovarianceMatrices.aVar itself.
@noinline residualadjustment(k::CM.CR0, r::OLSEstimator) = 1.0
@noinline residualadjustment(k::CM.CR1, r::OLSEstimator) = 1.0

function residualadjustment(k::CM.CR2, r::OLSEstimator)
    wts = r.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u_orig = residuals(r)
    u = copy(u_orig)
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)
    for groups in 1:g.ngroups
        ind = findall(x -> x .== groups, g)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false).L, ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= matrixpowbysvd(I - Hᵧᵧ, -0.5)*ug
        end
    end
    # Return the adjustment factor: adjusted_u / original_u
    # So that M = (X .* u_orig) .* factor = X .* adjusted_u
    return u ./ u_orig
end

function matrixpowbysvd(A, p; tol = eps()^(1/1.5))
    s = svd(A)
    V = s.S
    V[V .< tol] .= 0
    return s.V*diagm(0=>V .^ p)*s.Vt
end

function residualadjustment(k::CM.CR3, r::OLSEstimator)
    wts = r.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u_orig = residuals(r)
    u = copy(u_orig)
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)
    for groups in 1:g.ngroups
        ind = findall(g .== groups)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false), ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= (I - Hᵧᵧ)^(-1)*ug
        end
    end
    # Return the adjustment factor: adjusted_u / original_u
    return u ./ u_orig
end

function CM.vcov(k::CM.AbstractAsymptoticVarianceEstimator, m::OLSEstimator; dofadjust = true, kwargs...)
    A = aVar(k, m; kwargs...)
    n = nobs(m)
    B = invchol(m.pp)
    basis_coef = m.basis_coef

    # The aVar function returns M'M/n (where M is the adjusted moment matrix).
    # For HR1, residualadjustment = √(n/dof_residual), so the adjusted moment matrix
    # already incorporates part of the DOF adjustment.
    #
    # For HC0/HR0: V = n * B * (M'M/n) * B = B * M'M * B
    # For HC1/HR1: V = n/(n-k-k_fe) * B * M'M * B  (with proper DOF)
    # For cluster-robust: Similar, accounting for all absorbed DOF

    scale = if k isa Union{CM.HC1, CM.HR1}
        # HC1: DOF adjustment should account for both k and k_fe
        # residualadjustment for HR1 already applied √(n/dof_residual)
        # So aVar returns M'M * (n/dof_residual) / n = M'M / dof_residual
        # We want final scale = n / (n - k - k_fe)
        # With aVar = M'M / dof_residual and dof_residual ≈ n - k - k_fe - 1,
        # we need: scale * (1/dof_residual) = n / (n - k - k_fe)
        # => scale = n * dof_residual / (n - k - k_fe)
        p_total = dof(m) + dof_fes(m)
        n * dof_residual(m) / (n - p_total)
    elseif k isa Union{CM.CR0, CM.CR1, CM.CR2, CM.CR3}
        # Cluster-robust: same DOF adjustment
        p_total = dof(m) + dof_fes(m)
        n * dof_residual(m) / (n - p_total)
    else
        # HC0/HR0: no DOF adjustment, scale = n
        convert(eltype(A), n)
    end

    # Handle dimension mismatch when there is collinearity:
    # - A is k×k (full size, with NaN for collinear entries)
    # - B is k_reduced×k_reduced (from factorization on non-collinear columns)
    # We need to extract the valid submatrix, compute sandwich, then expand back
    if !all(basis_coef)
        k_full = length(basis_coef)
        valid_idx = findall(basis_coef)

        # Extract valid submatrix of A
        A_valid = A[valid_idx, valid_idx]

        # Compute sandwich on reduced dimensions
        Σ_valid = scale .* B * A_valid * B

        # Expand back to full size with NaN for collinear entries
        T = eltype(Σ_valid)
        Σ = fill(T(NaN), k_full, k_full)
        Σ[valid_idx, valid_idx] = Σ_valid

        return Σ
    end

    Σ = scale .* B * A * B

    return Σ
end
