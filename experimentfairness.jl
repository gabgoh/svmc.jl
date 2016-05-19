include("svm.jl")
include("loaddata.jl")

using DataFrames

# ───────────────────────────────────────────────────────────────────
# Setup Fairness Constraints
# ───────────────────────────────────────────────────────────────────

y = -y
ytest = -ytest

e = ones(m)

Male = (A[:,39] .== 1)[:]
NM = sum(Male)
Female = !Male
NF = sum(Female)

Malet = (Atest[:,39] .== 1)[:]
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

params = Dict{Any,Any}()

# ───────────────────────────────────────────────────────────────────
# Ramp
# ───────────────────────────────────────────────────────────────────

function eval_fairness(pred)
  
  R(x)  = min(max(1 - x, 0), 2)/2

  p = pred(A)
  pt = pred(Atest)

  M = sum(sign(p[Male])   .== -1)/NM
  F = sum(sign(p[Female]) .== -1)/NF

  Mtest = sum(sign(pt[Malet])  .== -1)/NMt
  Ftest = sum(sign(pt[Femalet]) .== -1)/NFt

  Acc  = sum(abs(y - sign(p)))/(2*length(p))
  Acct = sum(abs(ytest - sign(pt)))/(2*length(pt))

  RM = sum(R(p[Male]))/NM 
  RF = sum(R(p[Female]))/NF

  RMtest = sum(R(pt[Malet]))/NMt
  RFtest = sum(R(pt[Femalet]))/NFt

  # Evaluate ratios of randomized classifier

  randclass(p) = rand(length(p)) .< R(p)

  n_samples = 1000

  RandRatiotest = 0
  for i = 1:n_samples
    C = randclass(pt)
    RandRatiotest = RandRatiotest + sum(C[Malet])/sum(C[Femalet])
  end
  RandRatiotest = RandRatiotest/n_samples

  RandRatio = 0
  for i = 1:n_samples
    C = randclass(p)
    RandRatio = RandRatio + sum(C[Male])/sum(C[Female])
  end
  RandRatio = RandRatio/n_samples

  return Dict{Symbol, Float64}(:M => M, 
                               :F => F, 
                               :Mtest => Mtest, 
                               :Ftest => Ftest, 
                               :RM => RM,
                               :RF => RF,
                               :RMtest => RMtest,
                               :RFtest => RFtest,
                               :RandRatiotest => RandRatiotest,
                               :RandRation => RandRatio,
                               :Acc => Acc,
                               :Acct => Acct)

end

ConstraintRange = linspace(1,6.5,32*3)

#ConstraintRange = linspace(5,5 + (4/3),32)

function fairness_ramp_ratio(u)

  tic()

  yc = [ones(NM); -ones(NF)]
  Ac = [A[Male,:]; A[Female,:]]
  wc = [ones(NM)/(2*u*NM); ones(NF)/(2*NF)]

  try

    (pred, v, λ) = svmramp(y,A,e,yc,Ac,wc; 
                           solver = :liblinear)
    D            = eval_fairness(pred)
    D[:runtime]  = toc()
    D[:λ]        = λ
    D[:u]        = u

    return (D, v)

  end

  return nothing

end


function fairness_ramp_additive(c)

  tic()

  yc = [ones(NM); -ones(NF)]
  Ac = [A[Male,:]; A[Female,:]]
  η  = (c*(NM+NF))/(2*NM*NF) + 1
  wc = [ones(NM)/(2*NM*η); ones(NF)/(2*NF*η)]

  try

    (pred, v, λ) = svmramp(y,A,e,yc,Ac,wc; 
                           solver = :liblinear)
    D            = eval_fairness(pred)
    D[:runtime]  = toc()
    D[:λ]        = λ
    D[:c]        = c

    return (D, v)

  end

  return nothing

end

ZvrgConstraintRange = linspace(1,length(y)/2,32*3)

function fairness_zvrg(c; method = 1)
  
  tic()

  xf = (NF*sum(A[Male,:],1) - NM*sum(A[Female,:],1))/(NM + NF)

  (pred,v) = svmc_bisect(y,A,e,[1],xf,[1/(1+c)]; 
                         solver = :liblinear, 
                         verbose = true,
                         tol = 0.1)

  D            = eval_fairness(pred)
  D[:runtime]  = toc()
  D[:xf]       = pred(xf)[1]
  D[:λ]        = λ

  return (D, v, pred)

end

ConstraintRange = linspace(0,8000,100)

function save_run(z, f1, f2)

  D = DataFrame(; [symbol(k)=>v for (k,v) in z[1][1]]...)
  for i = 2:length(z)
    push!(D, [symbol(k)=>v for (k,v) in z[i][1]] )
  end

  SVL = spzeros(length(z[end][2]),0)
  for i = 1:length(z)
    if typeof(z[i][2]) != Float64
      SVi = z[i][2]
      SVi[abs(SVi) .< 1e-6] = 0
      SVL = [SVL sparse(SVi)]
    end
  end

  writetable(f1, D)
  writedlm(f2, full(SVL))
  
  return (D,SVL)

end
