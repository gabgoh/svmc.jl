
srand(1)
n           = 4
d           = 10000
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

@time (x,ρ,v) = svmc(y, A, w, ones(size(y)), A, w)
@time (x1,ρ1,v1) = svm(y, A, w);

  
srand(1)
n           = 4
d           = 1000
Red         = 1:Integer(d)
Blu         = (Integer(d)+1):(2*d)
m           = 2*d
y           = [ones(length(Red)); -ones(length(Blu))]
A           = 1*[randn(d,n); randn(d,n)]
A[Red,2]    = A[Red,2] + 4
A[Red,1]    = A[Red,1] + 3
A[Blu,2]    = A[Blu,2] + 5
w           = ones(2*d)

(x,ρ,v) = svmcbisect(y, A, w, ones(size(y)), A, w/500)

σ = sum(max( 1 - (A*x - ρ)[:], 0))
println(σ)
