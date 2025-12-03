"""
Predictor components for OLS estimation.

Similar to GLM.jl's DensePredChol and DensePredQR, these types store
the model matrix, coefficients, and factorizations for linear regression.
"""

abstract type OLSLinearPredictor{T <: AbstractFloat} end

"""
    OLSPredictorChol{T}

OLS predictor using Cholesky factorization of X'X.
Faster than QR but less numerically stable for ill-conditioned problems.
"""
mutable struct OLSPredictorChol{T <: AbstractFloat} <: OLSLinearPredictor{T}
    X::Matrix{T}                    # Model matrix (post-FE, post-weights)
    beta::Vector{T}                 # Coefficient estimates
    chol::Cholesky{T, Matrix{T}}    # Cholesky factorization of X'X
    delbeta::Vector{T}              # Scratch space for coefficient updates
    scratchm1::Matrix{T}            # Scratch matrix for computations
end

"""
    OLSPredictorQR{T}

OLS predictor using QR factorization of X.
More numerically stable than Cholesky but about 2x slower.
"""
mutable struct OLSPredictorQR{T <: AbstractFloat} <: OLSLinearPredictor{T}
    X::Matrix{T}                    # Model matrix
    beta::Vector{T}                 # Coefficient estimates
    qr::LinearAlgebra.QRCompactWY{T, Matrix{T}}   # QR factorization
    delbeta::Vector{T}              # Scratch space
    scratchm1::Matrix{T}            # Scratch matrix
end

"""
    linpred_rank(pp::OLSLinearPredictor) -> Int

Return the rank of the linear predictor (number of non-collinear coefficients).
"""
linpred_rank(pp::OLSLinearPredictor) = sum(.!isnan.(pp.beta))

"""
    invchol(pp::OLSPredictorChol) -> Symmetric

Compute (X'X)^(-1) from Cholesky factorization.
This is the "bread" of the sandwich variance estimator.
"""
function invchol(pp::OLSPredictorChol{T}) where {T}
    return Symmetric(inv(pp.chol))
end

"""
    invchol(pp::OLSPredictorQR) -> Symmetric

Compute (X'X)^(-1) from QR factorization.
Uses the relationship (X'X)^(-1) = (R'R)^(-1) = R^(-1) R^(-T).
"""
function invchol(pp::OLSPredictorQR{T}) where {T}
    R = pp.qr.R
    R_inv = inv(UpperTriangular(R))
    return Symmetric(R_inv * R_inv')
end

"""
    coefmatrix(pp::OLSLinearPredictor) -> Matrix

Return X'X for the predictor.
"""
function coefmatrix(pp::OLSPredictorChol{T}) where {T}
    return Matrix(pp.chol)
end

function coefmatrix(pp::OLSPredictorQR{T}) where {T}
    R = pp.qr.R
    return R' * R
end
