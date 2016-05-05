# Star compute cloud
using PyPlot

addprocs([("104.197.156.114",1),
          ("104.197.119.208",4),
          ("104.197.188.204",32)], 
          sshflags = `-i /Users/gabe/.ssh/my-ssh-key`, 
          dir = "/home/gabe/svmc/", 
          exename = "/usr/bin/julia", 
          tunnel=true)

@everywhere include("/Users/gabe/Dropbox/goh/svmc/experiment.jl")

ConstraintRange = linspace(1,2.0,37*4)
z = pmap(ramp_experiment, ConstraintRange)

F  = [i[:F] for i = z]
M  = [i[:M] for i = z]
Ft = [i[:Ft] for i = z]
Mt = [i[:Mt] for i = z]

plot(M,"k")
plot(Mt,"k--")
plot(F,"r")
plot(Ft,"r--")

title("Fairness Constraints")
xlabel("Fairness")
ylabel("Proportions")

legend(["Male (Train)", "Male (Test)", "Female (Train)", "Female (Test)"])