##############################################################################
##
## Kleibergen-Paap Rank Test for Weak Instruments
##
## Ported from Vcov.jl to work with CovarianceMatrices.jl
## Reference: Kleibergen & Paap (2006), "Generalized Reduced Rank Tests
##            Using the Singular Value Decomposition"
##
##############################################################################

using LinearAlgebra

"""
    ranktest(Xendo_res, Z_res, Pi, vcov_type, nobs, dof_small, dof_fes)

Compute the Kleibergen-Paap rk statistic for testing weak instruments.

# Arguments
- `Xendo_res::Matrix`: Residualized endogenous variables (n × k)
- `Z_res::Matrix`: Residualized instruments (n × l)
- `Pi::Matrix`: First-stage coefficient matrix
- `vcov_type`: Variance estimator type (from CovarianceMatrices.jl)
- `nobs::Int`: Number of observations
- `dof_small::Int`: Degrees of freedom (number of parameters)
- `dof_fes::Int`: Degrees of freedom absorbed by fixed effects

# Returns
- `r_kp`: The Kleibergen-Paap rk statistic
"""
function ranktest(
    Xendo_res::Matrix{T},
    Z_res::Matrix{T},
    Pi::Matrix{T},
    vcov_type,
    nobs::Int,
    dof_small::Int,
    dof_fes::Int
) where T<:AbstractFloat

    k = size(Xendo_res, 2)  # Number of endogenous variables
    l = size(Z_res, 2)      # Number of excluded instruments

    # Handle edge cases
    if k == 0 || l == 0
        return T(NaN)
    end

    # Compute theta = F * Pi * G' where F and G are Cholesky factors
    # This transforms the problem to have identity covariance
    ZZ = Symmetric(Z_res' * Z_res)
    XX = Symmetric(Xendo_res' * Xendo_res)

    Fmatrix_chol = cholesky(ZZ; check = false)
    Gmatrix_chol = cholesky(XX; check = false)

    if !issuccess(Fmatrix_chol) || !issuccess(Gmatrix_chol)
        return T(NaN)
    end

    Fmatrix = Fmatrix_chol.U
    Gmatrix = Gmatrix_chol.U

    # theta = F * Pi' * inv(G')
    theta = Fmatrix * (Gmatrix' \ Pi')'

    # Compute SVD decomposition
    svddecomp = svd(theta, full = true)
    u = svddecomp.U
    vt = svddecomp.Vt

    # Extract submatrices for the rank test (see Kleibergen-Paap p.102)
    u_sub = u[k:l, k:l]
    vt_sub = vt[k, k]

    # Compute a_qq and b_qq
    if iszero(u_sub)
        a_qq = u[1:l, k:l]
    else
        a_qq = u[1:l, k:l] * (u_sub \ sqrt(u_sub * u_sub'))
    end

    if iszero(vt_sub)
        b_qq = vt[1:k, k]'
    else
        b_qq = sqrt(vt_sub * vt_sub') * (vt_sub' \ vt[1:k, k]')
    end

    # Kronecker product for the test statistic
    kronv = kron(b_qq, a_qq')
    lambda = kronv * vec(theta)

    # Compute variance depending on vcov type
    if vcov_type isa CovarianceMatrices.HR0 || vcov_type isa CovarianceMatrices.HR1
        # Simple/homoskedastic case
        vlab_chol = cholesky(Hermitian((kronv * kronv') ./ nobs); check = false)
        if !issuccess(vlab_chol)
            return T(NaN)
        end
        r_kp = lambda' * (vlab_chol \ lambda)
    else
        # Robust/cluster case - compute sandwich variance
        K = kron(Gmatrix, Fmatrix)'

        # Compute the "meat" using CovarianceMatrices.jl
        # For robust inference, we need the moment conditions
        # M = Z_res ⊗ (Xendo_res - Z_res * Pi') adjusted for vcov type

        # The meat is E[mm'] where m is the vectorized moment condition
        # For now, use a simplified approach based on the original Vcov.jl

        # Compute residuals from first stage
        residuals_fs = Xendo_res - Z_res * Pi'

        # Build moment matrix: kron of Z with residuals columns
        n = nobs
        kl = k * l
        moment_matrix = zeros(T, n, kl)
        for j in 1:k
            for i in 1:l
                idx = (j - 1) * l + i
                moment_matrix[:, idx] = Z_res[:, i] .* residuals_fs[:, j]
            end
        end

        # Compute meat using CovarianceMatrices
        meat = _compute_meat(moment_matrix, vcov_type, nobs, dof_small, dof_fes)

        # Transform variance
        vhat = K \ (K \ meat)'
        vlab_matrix = Hermitian(kronv * vhat * kronv')
        vlab_chol = cholesky(vlab_matrix; check = false)

        if !issuccess(vlab_chol)
            return T(NaN)
        end
        r_kp = lambda' * (vlab_chol \ lambda)
    end

    return r_kp[1]
end

"""
    _compute_meat(moment_matrix, vcov_type, nobs, dof_small, dof_fes)

Compute the meat of the sandwich estimator for the rank test.
"""
function _compute_meat(
    moment_matrix::Matrix{T},
    vcov_type,
    nobs::Int,
    dof_small::Int,
    dof_fes::Int
) where T<:AbstractFloat

    n = size(moment_matrix, 1)

    if vcov_type isa CovarianceMatrices.HR0
        # No adjustment
        return moment_matrix' * moment_matrix
    elseif vcov_type isa CovarianceMatrices.HR1
        # HC1 adjustment
        dof_residual = max(1, n - dof_small - dof_fes)
        scale = n / dof_residual
        return scale * (moment_matrix' * moment_matrix)
    elseif vcov_type isa Union{CovarianceMatrices.CR0, CovarianceMatrices.CR1}
        # Cluster-robust
        clusters = vcov_type.g[1]
        n_clusters = length(unique(clusters))

        # Sum moments within clusters
        cluster_sums = zeros(T, n_clusters, size(moment_matrix, 2))
        for (i, c) in enumerate(clusters)
            cluster_sums[c, :] .+= moment_matrix[i, :]
        end

        meat = cluster_sums' * cluster_sums

        # Apply small-sample correction for CR1
        if vcov_type isa CovarianceMatrices.CR1
            dof_residual = max(1, n - dof_small - dof_fes)
            scale = (n_clusters / (n_clusters - 1)) * (n / dof_residual)
            meat *= scale
        end

        return meat
    else
        # Default: use HC1-like computation
        dof_residual = max(1, n - dof_small - dof_fes)
        scale = n / dof_residual
        return scale * (moment_matrix' * moment_matrix)
    end
end

"""
    compute_first_stage_fstat(Xendo_res, Z_res, Pi, vcov_type, nobs, dof_small, dof_fes)

Compute the Kleibergen-Paap first-stage F-statistic and p-value.

Returns (F_kp, p_kp) where:
- F_kp is the first-stage F-statistic
- p_kp is the p-value from chi-squared distribution
"""
function compute_first_stage_fstat(
    Xendo_res::Matrix{T},
    Z_res::Matrix{T},
    Pi::Matrix{T},
    vcov_type,
    nobs::Int,
    dof_small::Int,
    dof_fes::Int
) where T<:AbstractFloat

    k = size(Xendo_res, 2)  # Number of endogenous variables
    l = size(Z_res, 2)      # Number of excluded instruments

    try
        r_kp = ranktest(Xendo_res, Z_res, Pi, vcov_type, nobs, dof_small, dof_fes)

        if isnan(r_kp)
            return T(NaN), T(NaN)
        end

        # Degrees of freedom for chi-squared test
        df = l - k + 1

        # P-value from chi-squared distribution
        p_kp = chisqccdf(df, r_kp)

        # F-statistic: divide by number of instruments
        F_kp = r_kp / l

        return F_kp, p_kp
    catch e
        @info "ranktest failed: $e; first-stage statistics not estimated"
        return T(NaN), T(NaN)
    end
end
