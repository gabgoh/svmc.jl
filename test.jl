function genData()

  srand(1)
  n           = 4
  d           = 100
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

  return (y,A,w)

end

function test1()

  (y,A,w)    = genData()
  (pred,v)   = svmc(y, A, w, ones(size(y)), A, w)
  (pred,v1)  = svm(y, A, w);

end

function test2()  

  (y,A,w) = genData()
  (pred,v) = svmcbisect(y, A, w, ones(size(y)), A, w/500)
  (pred,v) = svmc(y, A, w, ones(size(y)), A, w/500)

end