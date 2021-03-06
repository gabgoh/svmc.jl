# Support Vector Machines with Dataset Constraints

Solves the ramp constrained optimization problem
```
minimizeₓ   w₀ᵀmin{max{1 - Y₀(A₀x - b)},2} + ½xᵀx
s.t.        wᵢᵀmin{max{1 - Yᵢ(Aᵢx - b)},2} ≦ 1
```

And the hinge constrained optimization problem
```
minimizeₓ     w₀ᵀmax(1 - Y₀(A₀x - b), 0) + ½xᵀx
s.t.          wᵢᵀmax(1 - Yᵢ(Aᵢx - b), 0) ≦ 1
```

## Example - Neyman-Pearson Learning

```
minimize    False Positives + ½xᵀx
s.t.        False Negatives ≦ 10
```

The Julia code is as follows

```julia
# Load SVM solver
include("svm.jl")

# Generate Synthetic Data, y (labels), A (data), w (data-weights)
include("loaddata.jl")

# Train SVM Model (Ramp Loss)
y₊ = y .> 0; y₋ = y .< 0
(pred, v, λ) = svmramp( y[y₊], A[y₊,:], e[y₊],    
                        y[y₋], A[y₋,:], e[y₋]/10 )

# Get Predictions
pr = sign(pred(A))

# Train SVM Model (Hinge Loss)
(pred, v, λ) = svmc_bisect( y[y₊], A[y₊,:], e[y₊],    
                            y[y₋], A[y₋,:], e[y₋]/10 )

# Get Predictions
ph = sign(pred(A))
```
