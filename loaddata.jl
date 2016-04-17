
using LIBSVM

dataset = 1
toplot = false

# Adult Dataset
if dataset == 1
  (y,A)         = parse_libSVM("data/a1a")
  (ytest,Atest) = parse_libSVM("data/a1a.t")

  n = max(size(A,1), size(Atest,1))
  m = size(A,2)
  A = [A; spzeros(n-size(A,1),size(A,2))]
  Atest = [Atest; spzeros(n-size(Atest,1), size(Atest,2))]

  m = size(A,2)
  A = A'
  Atest = Atest'
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
