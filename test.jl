include("svm.jl")

using Base.Test

srand(1)
n           = 2
d           = 200
Red         = 1:Integer(d)
Blu         = (Integer(d)+1):(2*d)
m           = 2*d
y           = [ones(length(Red)); -ones(length(Blu))]
A           = 1*[randn(d,n); randn(d,n)]
A[Red,2]    = A[Red,2] + 4
A[Red,1]    = A[Red,1] + 3
A[Blu,2]    = A[Blu,2] + 5
A           = 3*A
w           = rand(2*d)
e           = ones(2*d)

(pred,v1)  = svmc(y, A, w)
(pred,v2)  = svm(y, A, w)

@test norm(v1 - v2,Inf)/m <= 1e-5

# Make sure SVMs behave in sane ways w.r. to 0 weights

Id = 50:200

e1 = ones(size(e))
e1[Id] = e[Id]

(pred,v10)  = svmc(y, A, e1)
(pred,v20)  = svm(y, A, e1)

@test norm(v10 - v20)/m <= 1e-5

# Test constrained optimiztion

svmramp(y,A,e,e,A,e/45)

