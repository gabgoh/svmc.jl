# Star compute cloud
using PyPlot

addprocs([("104.197.156.114",1),
          ("104.197.119.208",4),
          ("104.197.188.204",32)], 
          sshflags = `-i /Users/gabe/.ssh/my-ssh-key`, 
          dir = "/home/gabe/svmc/", 
          exename = "/usr/bin/julia", 
          tunnel=true)

include("experiment.jl")

zz = pmap(hinge_experiment, ConstraintRange)

