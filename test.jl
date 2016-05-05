include("svm.jl")

using Base.Test

srand(1)
n           = 2
d           = 30
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
(pred,v2)  = svm_libsvm(y, A, w)

@test norm(v1 - v2,Inf)/m <= 1e-5

# Make sure SVMs behave in sane ways w.r. to 0 weights

Id = round(Int,d/2):m

e1 = zeros(size(e))
e1[Id] = e[Id]

(pred1,v10)  = svmc(y, A, e1)
(pred2,v20)  = svm_libsvm(y, A, e1)
(pred3,v30)  = svm_liblinear(y, A, e1)

@test norm(v10 - v20)/m <= 1e-5
@test norm(pred1(A) - pred2(A))/min(norm(pred1(A)),norm(pred2(A))) < 0.1
@test norm(sign(pred1(A)) - sign(pred2(A))) == 0 # Same predictions

(pred,v11)  = svmc(y[Id], A[Id,:], e1[Id])
(pred,v21)  = svm_libsvm(y[Id], A[Id,:], e1[Id])

@test norm(v10 - v20)/m <= 1e-5
@test norm(v11 - v10[Id])/m <= 1e-5
@test norm(v21 - v20[Id])/m <= 1e-5

# Test constrained optimiztion. Make sure the interior point and
# bisection methods give (roughly) the same answers.

(pred1, v1) = svmc(y,A,e,e,A,e/45)
(pred2, v2) = svmcbisect(y,A,e,e,A,e/45, 
                         tol = 1e-6,verbose = true, maxiters = 15)

@test norm(pred1(A) - pred2(A), Inf) < 0.01
@test sum(abs(sign(pred1(A)) - sign(pred2(A)))) == 0


