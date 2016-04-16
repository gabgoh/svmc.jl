using LIBSVM
using PyPlot
using ProgressMeter
using Debug
include("svm.jl")
include("loaddata.jl")

e  = ones(length(y))

y₊ = y .> 0
y₋ = y .< 0

ytest₊ = ytest .> 0
ytest₋ = ytest .< 0

# ───────────────────────────────────────────────────────────────────
# Setup SVM Parameters
# ───────────────────────────────────────────────────────────────────

ConstraintRange = 500 #linspace(1,3200,50)

params = Dict{Any,Any}( :kernel => Int32(2) , 
                        :degree => 2, 
                        :γ => 1/n )

# ───────────────────────────────────────────────────────────────────
# Ramp
# ───────────────────────────────────────────────────────────────────

ramp  = DataFrame(fp = Float64[], fn  = Float64[])
#vramp = zeros(m,0)

for η = ConstraintRange

  (pred, v) = svmramp( y[y₊], A[y₊,:], e[y₊],
                       y[y₋], A[y₋,:], e[y₋]/η ;
                       params...)

  (err, fp, fn, tp, tn) = calc_error(Atest', ytest, a -> sign(pred(a'))[1]  )

  println( η, "\t", err, "\t", fp, "\t", fn)

  push!(ramp, [fp fn])
  #vramp = [vramp v]

end

# ───────────────────────────────────────────────────────────────────
# Hinge
# ───────────────────────────────────────────────────────────────────

hinge = DataFrame(fp = Float64[], fn  = Float64[])

for η = ConstraintRange

  (pred, v, λ) = svmcbisect( y[y₊], A[y₊,:], e[y₊],
                             y[y₋], A[y₋,:], e[y₋]/η ; 
                             verbose = true, params ... )

  #(err, fp, fn, tp, tn) = calc_error(Atest', ytest, a -> sign(pred(a'))[1] )
  (err, fp, fn, tp, tn) = calc_error(A', y, a -> sign(pred(a'))[1] )

  println( err, "\n", fp, "\n", fn , "\n" , λ)

  push!(hinge, [fp fn])

end

# ───────────────────────────────────────────────────────────────────
# Bias Shift
# ───────────────────────────────────────────────────────────────────

(pred, v) = ksvm(y, A, e; params...)

z = pred(Atest)
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

# ───────────────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────────────

figure()

plot( hinge[:fn]     , hinge[:fp]     , "k" )
plot( ramp[:fn]      , ramp[:fp]      , "g" )
plot( biasshift[:fn] , biasshift[:fp] , "r" )

# ───────────────────────────────────────────────────────────────────
# 2D Plot
# ───────────────────────────────────────────────────────────────────

if false

figure()

plot( A[ y .== -1 , 1] , A[ y .== -1 , 1],"r." )
plot( A[ y .==  1 , 1] , A[ y .==  1 , 1],"b." )

decbound(y)  = (ρ - x[1]*y)/x[2]
decbound2(y) = (-1 + ρ - x[1]*y)/x[2]

ax = axis();
plot( [ ax[1]; ax[2] ], [ decbound(ax[1])  ; decbound(ax[2])  ] , "k"   )
plot( [ ax[1]; ax[2] ], [ decbound2(ax[1]) ; decbound2(ax[2]) ] , "k--" )

axis(ax);

end
