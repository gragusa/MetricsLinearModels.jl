"""
Core OLS solver utilities for MetricsLinearModels.jl

This file contains functions for:
- Collinearity detection
- Building predictor objects with factorizations
- Solving least squares problems
- NaN expansion for rank-deficient cases
"""

using LinearAlgebra
using LinearAlgebra: BlasReal

"""
    detect_collinearity(X, has_intercept; tol) -> BitVector

Detect collinear columns in design matrix X.

Uses QR factorization with column pivoting to identify linearly dependent columns.
Returns a BitVector where `true` indicates the column is in the basis (non-collinear).

# Arguments
- `X::Matrix{T}`: Design matrix
- `has_intercept::Bool`: Whether model has an intercept
- `tol::Real=1e-8`: Tolerance for detecting near-zero pivots

# Returns
- `BitVector`: Indicator of non-collinear columns
"""
function detect_collinearity(X::Matrix{T}, has_intercept::Bool;
        tol::Real = 1e-8) where {T <: AbstractFloat}
    n, k = size(X)
    basis = trues(k)

    # Quick check: zero columns
    for j in 1:k
        col_norm = norm(view(X, :, j))
        if col_norm < tol
            basis[j] = false
        end
    end

    # If all columns are zero, return early
    if !any(basis)
        return basis
    end

    # Use QR with column pivoting to detect linear dependence
    X_nonzero = X[:, basis]
    F = qr(X_nonzero, ColumnNorm())

    # Check diagonal of R for near-zero elements
    R_diag = abs.(diag(F.R))
    max_diag = maximum(R_diag)
    threshold = max_diag * tol

    # Find rank
    rank = sum(R_diag .> threshold)

    # Mark collinear columns
    if rank < size(X_nonzero, 2)
        basis_indices = findall(basis)
        for j in (rank + 1):length(R_diag)
            basis[basis_indices[F.p[j]]] = false
        end
    end

    return basis
end

"""
    expand_coef_nan(coef_reduced, basis) -> Vector

Expand reduced coefficient vector to full size with NaN for collinear columns.

# Arguments
- `coef_reduced::Vector{T}`: Coefficients for non-collinear columns only
- `basis::BitVector`: Indicator of non-collinear columns

# Returns
- `Vector{T}`: Full coefficient vector with NaN for collinear columns
"""
function expand_coef_nan(coef_reduced::Vector{T}, basis::BitVector) where {T}
    k = length(basis)
    coef_full = fill(T(NaN), k)
    coef_full[basis] = coef_reduced
    return coef_full
end

"""
    update_coef_nan!(coef_full, coef_reduced, basis)

Update full coefficient vector in-place with NaN for collinear columns.

# Arguments
- `coef_full::Vector{T}`: Full coefficient vector to update
- `coef_reduced::Vector{T}`: Coefficients for non-collinear columns
- `basis::BitVector`: Indicator of non-collinear columns
"""
function update_coef_nan!(coef_full::Vector{T}, coef_reduced::Vector{T},
        basis::BitVector) where {T}
    coef_full[basis] = coef_reduced
    coef_full[.!basis] .= T(NaN)
    return coef_full
end

"""
    expand_vcov_nan(vcov_reduced, basis) -> Symmetric

Expand reduced vcov matrix to full size with NaN for collinear columns.

# Arguments
- `vcov_reduced::Symmetric{T}`: Vcov for non-collinear columns
- `basis::BitVector`: Indicator of non-collinear columns

# Returns
- `Symmetric{T}`: Full vcov matrix with NaN rows/columns for collinear variables
"""
function expand_vcov_nan(vcov_reduced::Symmetric{T}, basis::BitVector) where {T}
    k = length(basis)
    vcov_full = fill(T(NaN), k, k)
    vcov_full[basis, basis] = Matrix(vcov_reduced)
    return Symmetric(vcov_full)
end

"""
    build_predictor(X, y, factorization, has_intercept) -> (predictor, basis_coef)

Build predictor object with factorization and detect collinearity.

# Arguments
- `X::Matrix{T}`: Design matrix
- `y::Vector{T}`: Response vector (needed for solving)
- `factorization::Symbol`: `:chol` or `:qr`
- `has_intercept::Bool`: Whether model has intercept

# Returns
- Predictor object (OLSPredictorChol or OLSPredictorQR)
- `basis_coef::BitVector`: Indicator of non-collinear coefficients
"""
function build_predictor(X::Matrix{T}, y::Vector{T},
        factorization::Symbol,
        has_intercept::Bool) where {T <: AbstractFloat}
    n, k = size(X)

    # Detect collinearity
    basis_coef = detect_collinearity(X, has_intercept)

    # Remove collinear columns
    X_reduced = X[:, basis_coef]
    k_reduced = size(X_reduced, 2)

    if factorization == :chol
        # Compute X'X and Cholesky factorization
        XX = Symmetric(X_reduced' * X_reduced)
        chol_fact = cholesky(XX)

        # Solve for coefficients: β = (X'X)^(-1) X'y
        Xy = X_reduced' * y
        beta_reduced = chol_fact \ Xy

        # Initialize scratch space
        delbeta = zeros(T, k_reduced)
        scratchm1 = zeros(T, k_reduced, k_reduced)

        # Expand coefficients with NaN for collinear columns
        beta = expand_coef_nan(beta_reduced, basis_coef)

        pp = OLSPredictorChol(X, beta, chol_fact, delbeta, scratchm1)

    elseif factorization == :qr
        # Compute QR factorization
        qr_fact = qr(X_reduced)

        # Solve for coefficients: β = R^(-1) Q'y
        beta_reduced = qr_fact \ y

        # Initialize scratch space
        delbeta = zeros(T, k_reduced)
        scratchm1 = zeros(T, n, k_reduced)

        # Expand with NaN
        beta = expand_coef_nan(beta_reduced, basis_coef)

        pp = OLSPredictorQR(X, beta, qr_fact, delbeta, scratchm1)
    else
        error("factorization must be :chol or :qr, got :$factorization")
    end

    return pp, basis_coef
end

"""
    solve_ols!(rr, pp, basis_coef)

Solve OLS problem and update response fitted values.

Updates `pp.beta` and `rr.mu` in place.

# Arguments
- `rr::OLSResponse{T}`: Response object
- `pp::OLSLinearPredictor{T}`: Predictor object (Chol or QR)
- `basis_coef::BitVector`: Indicator of non-collinear coefficients

# Returns
- `beta::Vector{T}`: Updated coefficient vector
"""
function solve_ols!(rr::OLSResponse{T},
        pp::OLSPredictorChol{T},
        basis_coef::BitVector) where {T}
    # Get reduced X
    X_reduced = pp.X[:, basis_coef]

    # Solve using Cholesky: β = (X'X)^(-1) X'y
    Xy = X_reduced' * rr.y
    beta_reduced = pp.chol \ Xy

    # Update full coefficient vector (with NaN for collinear)
    update_coef_nan!(pp.beta, beta_reduced, basis_coef)

    # Compute fitted values: mu = X * beta
    mul!(rr.mu, X_reduced, beta_reduced)

    return pp.beta
end

function solve_ols!(rr::OLSResponse{T},
        pp::OLSPredictorQR{T},
        basis_coef::BitVector) where {T}
    # Get reduced X
    X_reduced = pp.X[:, basis_coef]

    # Solve using QR: β = R^(-1) Q'y
    beta_reduced = pp.qr \ rr.y

    # Update coefficients
    update_coef_nan!(pp.beta, beta_reduced, basis_coef)

    # Compute fitted values
    mul!(rr.mu, X_reduced, beta_reduced)

    return pp.beta
end

"""
    compute_crossproduct(X) -> Symmetric

Compute X'X efficiently using BLAS syrk (symmetric rank-k update).

# Arguments
- `X::Matrix{T}`: Design matrix

# Returns
- `Symmetric{T}`: X'X as a symmetric matrix
"""
function compute_crossproduct(X::Matrix{T}) where {T <: BlasReal}
    k = size(X, 2)
    XX = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'T', one(T), X, zero(T), XX)
    return Symmetric(XX, :U)
end

"""
    compute_residuals!(residuals, y, X, coef) -> residuals

Compute residuals in-place: residuals = y - X*coef

# Arguments
- `residuals::Vector{T}`: Preallocated residuals vector
- `y::Vector{T}`: Response vector
- `X::Matrix{T}`: Design matrix
- `coef::Vector{T}`: Coefficient vector

# Returns
- `residuals::Vector{T}`: Updated residuals vector
"""
function compute_residuals!(residuals::Vector{T}, y::Vector{T},
        X::Matrix{T}, coef::Vector{T}) where {T}
    copyto!(residuals, y)
    # Remove collinear (NaN) coefficients
    valid = .!isnan.(coef)
    if any(valid)
        BLAS.gemv!('N', -one(T), X[:, valid], coef[valid], one(T), residuals)
    end
    return residuals
end
