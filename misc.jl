function calc_error(A, y, predict)

  err = 0; fp  = 0; fn  = 0; tp = sum(y.==1); tn = sum(y.==-1)

  c = sign(predict(A))[:]

  for i = 1:size(A,1)
    if abs(c[i] - y[i]) != 0
      err = err + 1
      if c[i] == 1 && y[i] == -1;
        fp = fp + 1
      else
        fn = fn + 1
      end
    end
  end

  return (err, fp, fn, tp, tn)

end

function Ind(I)

  z = zeros(length(I))
  for i = 1:length(I)
    z[i] = I[i] ? 1 : 0
  end
  return z

end
