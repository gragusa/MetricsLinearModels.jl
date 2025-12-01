##############################################################################
##
## IV Estimator Types
##
##############################################################################

"""
    AbstractIVEstimator

Abstract type for instrumental variables estimators.

Concrete subtypes include:
- `TSLS`: Two-Stage Least Squares
- `LIML`: Limited Information Maximum Likelihood (not yet implemented)
"""
abstract type AbstractIVEstimator end

"""
    TSLS <: AbstractIVEstimator

Two-Stage Least Squares (2SLS) estimator for instrumental variables models.

This is the most common IV estimator, which proceeds in two stages:
1. First stage: Regress endogenous variables on instruments and exogenous variables
2. Second stage: Regress outcome on predicted endogenous variables and exogenous variables

# Usage
```julia
iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))
```
"""
struct TSLS <: AbstractIVEstimator end

"""
    LIML <: AbstractIVEstimator

Limited Information Maximum Likelihood estimator for instrumental variables models.

**Note**: This estimator is not yet implemented. Calling `iv(LIML(), ...)` will throw an error.

LIML is an alternative to 2SLS that can have better finite-sample properties,
especially when instruments are weak.

# Future Usage
```julia
iv(LIML(), df, @formula(y ~ x + (endo ~ instrument)))
```
"""
struct LIML <: AbstractIVEstimator end
