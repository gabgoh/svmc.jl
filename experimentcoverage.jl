include("svm.jl")
include("loaddata.jl")

# ───────────────────────────────────────────────────────────────────
# Setup Precision/Recall constraints
# ───────────────────────────────────────────────────────────────────

e  = ones(length(y))

y₊ = y .> 0
y₋ = y .< 0

ytest₊ = ytest .> 0
ytest₋ = ytest .< 0

# ───────────────────────────────────────────────────────────────────
# Setup SVM Parameters
# ───────────────────────────────────────────────────────────────────

ConstraintRange = linspace(1,size(A,1)*1.5,32)

params = Dict{Any,Any}( :kernel => Int32(1) , 
                        :degree => 2, 
                        :γ => 1. )

# params = Dict{Any,Any}( :kernel => Int32(2) , 
#                         :degree => 2, 
#                         :γ => 2/n )

# ───────────────────────────────────────────────────────────────────
# Bias Shift
# ───────────────────────────────────────────────────────────────────

# (pred, v) = svm(y, A, e; params...)

# z = pred(Atest)[:]
# P = sortperm(z)

# biasshift = DataFrame(fp = Float64[], fn  = Float64[])

# for i = 1:100:size(P,1)

#   prd = zeros(size(P,1))
#   prd[P[i+1:end]] =  1
#   prd[P[1:i]]     = -1

#   err = abs(prd - ytest)/2
#   fp  = sum(err[ytest₋])
#   fn  = sum(err[ytest₊])

#   push!(biasshift, [fp fn])

# end

# ───────────────────────────────────────────────────────────────────
# Hinge
# ───────────────────────────────────────────────────────────────────

function hinge_experiment(η)
  
  tic()
  
  (pred, v, λ) = svmcbisect( y[y₊], A[y₊,:], e[y₊],
                             y[y₋], A[y₋,:], e[y₋]/η ; verbose = true, tol = 0.01, maxiters = 100,
                             params ... )

  (err, fp, fn, tp, tn) = calc_error(Atest, ytest, pred)

  println( η, "\t", err, "\t", fp, "\t", fn, "\t", λ)

  return Dict{Symbol, Any}(:fp => fp, :fn => fn, :v => v, :runtime => toc())

end

# ───────────────────────────────────────────────────────────────────
# Ramp
# ───────────────────────────────────────────────────────────────────

function ramp_experiment(η)

  tic()

  (pred, v) = svmramp( y[y₊], A[y₊,:], e[y₊],
                       y[y₋], A[y₋,:], e[y₋]/η ; verbose = true,
                       params...)

  (err, fp, fn, tp, tn) = calc_error(Atest, ytest, pred)

  println( η, "\t", err, "\t", fp, "\t", fn)

  return Dict{Symbol, Any}(:fp => fp, :fn => fn, :v => v, :runtime => toc())

end
