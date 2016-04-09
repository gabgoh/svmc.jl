using LIBSVM
using PyPlot
using ProgressMeter
using Debug
include("svm.jl")

dataset = 1
toplot = false

# Real Dataset
if dataset == 1
  (y,A) = parse_libSVM("data/a1a")
  n = size(A,1)
  m = size(A,2)
  A = A'
end

# Fake Dataset
if dataset == 2
  srand(1)
  n           = 2
  d           = 1000
  Red         = 1:Integer(d)
  Blu         = (Integer(d)+1):(2*d)
  m           = 2*d
  y           = [ones(length(Red)); -ones(length(Blu))]
  A           = 1*[randn(d,n); randn(d,n)]
  A[Red,2]    = A[Red,2] + 4
  A[Red,1]    = A[Red,1] + 3
  A[Blu,2]    = A[Blu,2] + 5
  A[Red[1],1:2] = [-2 5]
  A[Blu[1:10],1] = A[Blu[1:10],1] - 10;    
end

e           = ones(length(y))
y₊          = y .> 0
y₋          = y .< 0

ConstraintRange = linspace(1,2000, 200)

ramp = DataFrame(fp = Float64[], fn  = Float64[])

for η = ConstraintRange

  (x, ρ, v) = svmramp( y[y₊], A[y₊,:], e[y₊], 
                       y[y₋], A[y₋,:], e[y₋]/η )

  (err, fp, fn, tp, tn) = calc_error(A', y, a -> sign(a'x - ρ)[1] )

  push!(ramp, [fp fn])

end

hinge = DataFrame(fp = Float64[], fn  = Float64[])

for η = ConstraintRange

  (x, ρ, v) = svmc( y[y₊], A[y₊,:], e[y₊], 
                    y[y₋], A[y₋,:], e[y₋]/η)

  (err, fp, fn, tp, tn) = calc_error(A', y, a -> sign(a'x - ρ)[1] )

  push!(hinge, [fp fn])

end

biasshift = DataFrame(fp = Float64[], fn  = Float64[])

(x,ρ,v) = svmc(y, A, e)

for b = linspace(-5,5,1000)

  (err, fp, fn, tp, tn) = calc_error(A', y, a -> sign(a'x - ρ + b)[1] )
  push!(biasshift, [fp fn])
 
end

plot( hinge[:fn]     , hinge[:fp]     , "k" )
plot( ramp[:fn]      , ramp[:fp]      , "g" )
plot( biasshift[:fn] , biasshift[:fp] , "r" )

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