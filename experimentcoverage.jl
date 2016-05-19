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
                        :degree => 2        , 
                        :γ => 1.            ,
                        :solver => :libsvm )

# params = Dict{Any,Any}( :kernel => Int32(2) , 
#                         :degree => 2, 
#                         :γ => 2/n )

# ───────────────────────────────────────────────────────────────────
# Bias Shift
# ───────────────────────────────────────────────────────────────────

function biasshift()

  (pred, v) = svm(y, A, e; params...)

  z = pred(Atest)[:]
  P = sortperm(z)

  biasshift = DataFrame(fp = Float64[], fn  = Float64[])

  for i = 1:100:size(P,1)

    prd = zeros(size(P,1))
    prd[P[i+1:end]] =  1
    prd[P[1:i]]     = -1

    err = abs(prd - ytest)/2
    fp  = sum(err[ytest₋])
    fn  = sum(err[ytest₊])

    push!(biasshift, [fp fn])

  end

  return biasshift

end

# ───────────────────────────────────────────────────────────────────
# Hinge
# ───────────────────────────────────────────────────────────────────

function hinge_experiment(η)
  
  tic()
  
  (pred, v, λ) = svmc_bisect( y[y₊], A[y₊,:], e[y₊],
                              y[y₋], A[y₋,:], e[y₋]/η ;
                              verbose = true,
                              params ... )

  (errt, fpt, fnt, tpt, tnt) = calc_error(A, y, pred)
  (err,  fp,  fn,  tp,  tn ) = calc_error(Atest, ytest, pred)

  println( η, "\t", err, "\t", fp, "\t", fn, "\t", λ)

  return (Dict{Symbol, Any}(:fp => fp, 
                           :fn => fn, 
                           :fpt => fpt,
                           :fnt => fnt,
                           :η => η,
                           :λ => λ,
                           :runtime => toc()), v)

end

# ───────────────────────────────────────────────────────────────────
# Ramp
# ───────────────────────────────────────────────────────────────────

function ramp_experiment(η)

  tic()
  try
    (pred, v, λ) = svmramp( y[y₊], A[y₊,:], e[y₊],
                         y[y₋], A[y₋,:], e[y₋]/η ; verbose = true,
                         params...)

    (errt, fpt, fnt, tpt, tnt) = calc_error(A, y, pred)
    (err, fp, fn, tp, tn) = calc_error(Atest, ytest, pred)

    println( η, "\t", err, "\t", fp, "\t", fn)

    return (Dict{Symbol, Any}(:fp => fp, 
                             :fn => fn, 
                             :fpt => fpt,
                             :fnt => fnt,
                             :η => η,
                             :λ => λ,
                             :runtime => toc()), v)
  end

    return (Dict{Symbol, Any}(:fp => NaN, 
                             :fn => NaN, 
                             :fpt => NaN,
                             :fnt => NaN,
                             :η => NaN,
                             :λ => NaN,
                             :runtime => toc()), NaN)

end

# ───────────────────────────────────────────────────────────────────
# Save
# ───────────────────────────────────────────────────────────────────

function save(z)

  

end