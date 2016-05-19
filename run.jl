include("experimentfairness.jl")

ConstraintRange = linspace(1,6.5,32)

e1 = pmap(fairness_ramp_ratio, ConstraintRange)

ZvrgConstraintRange = linspace(1,length(y)/2,32)

e2 = pmap(fairness_zvrg, ZvrgConstraintRange)