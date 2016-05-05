include("svm.jl")
include("loaddata.jl")

# ───────────────────────────────────────────────────────────────────
# Setup Fairness Constraints
# ───────────────────────────────────────────────────────────────────

y = -y
ytest = -ytest

e = ones(m)

Male = A[:,39] .== 1
NM = sum(Male)
Female = !Male
NF = sum(Female)

Malet = Atest[:,39] .== 1
NMt = sum(Malet)
Femalet = !Malet
NFt = sum(Femalet)

# Upper bound

# # Lower bound

# l = 0.8
# yc = [-ones(NM); ones(m)]
# Ac = [A[Male,:]; A]
# wc = [ones(NM)/(l*NM); ones(m)/(l*m)]

# ───────────────────────────────────────────────────────────────────
# Setup SVM Parameters
# ───────────────────────────────────────────────────────────────────

#(pred, v, λ) = svmc(y,A,e)

ConstraintRange = linspace(1,1.5,37*4)

params = Dict{Any,Any}()

# ───────────────────────────────────────────────────────────────────
# Ramp
# ───────────────────────────────────────────────────────────────────

function ramp_experiment(u)

  tic()

  yc = [ones(NM); -ones(NF)]
  Ac = [A[Male,:]; A[Female,:]]
  wc = [ones(NM)/(2*u*NM); ones(NF)/(2*NF)]

  (pred, v) = svmramp(y,A,e,yc,Ac,wc)

  p = pred(A)
  pt = pred(Atest)

  M = sum(sign(p[Male])   .== -1)/NM
  F = sum(sign(p[Female]) .== -1)/NF

  Mt = sum(sign(pt[Malet])  .== -1)/NMt
  Ft = sum(sign(pt[Femalet]) .== -1)/NFt

  Acc  = sum(abs(y - sign(p)))/(2*length(p))
  Acct = sum(abs(ytest - sign(pt)))/(2*length(pt))

  return Dict{Symbol, Any}(:M => M, 
                           :F => F, 
                           :Mt => Mt, 
                           :Ft => Ft, 
                           :v => v, 
                           :Acc => Acc,
                           :Acct => Acct,
                           :runtime => toc())

end
