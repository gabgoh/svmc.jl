using IntPoint
using LIBSVM
export svmc, svm
include("bisect.jl")
include("misc.jl")

function parseargs(y₀, A₀, w₀, P)

  if length(P) % 3 != 0
    throw("Invalid Arguments")
  end

  k = Int(length(P)/3)

  w_ = Array{Vector}(0)
  y_ = Array{Vector}(0)
  A_ = Array{Matrix}(0)
  m_ = Array{Integer}(0)
  push!(w_, w₀); push!(A_, A₀); push!(y_, y₀); push!(m_, size(A₀,1))

  for i = 1:(length(P)/3)   
    push!(y_, P[3(i - 1) + 1]) 
    push!(A_, P[3(i - 1) + 2])
    push!(w_, P[3(i - 1) + 3])
    push!(m_, size(P[3(i - 1) + 2],1))
  end

  y     = sparse(cat(1, y_...)); 
  w     = sparse(cat(1, w_...)); 
  A     = cat(1, A_...)
  wc    = sparse(cat(1, zeros(m_[1],k), w_[2:end]...))

  (m,n) = size(A)
  k     = Int(length(P)/3)

  return (y_,A_,w_,m_)

end

# ───────────────────────────────────────────────────────────────────
# Given a svm y, A, w and primal solution xstar, recover bias value
# ───────────────────────────────────────────────────────────────────
function bias( y, A, w, xstar )

  z0 = 1 - y.*A*xstar  # the left part of 1 - YAx + Yρ
  yw = y.*w            # Potential gradient weights

  # Find all the discontinuties in the subgradient
  # of the objective function 
  #
  # f(ρ) = w'max{ 1 - Y(Ax - ρ) , 0 }
  #
  bpts = sort(((y.*A*xstar - 1)./y)[:])

  function ∂f(ρ)
    z = z0 + y.*ρ   
    I₊ = (z .> 0)[:]
    return sum(yw[I₊])
  end

  l = 1; u = length(bpts)

  # Binary search on breakpoints
  while true
    t = round(Int, l + (u - l)/2)
    val = ∂f(bpts[t])

    if val == 0; return bpts[t]; end
    val >= 0 ? u = t : l = t
    if u - l == 1
      return bpts[u]
    end
  end

end

# ───────────────────────────────────────────────────────────────────
# Constrained SVM, using IntPoint
# ───────────────────────────────────────────────────────────────────
function svmc(y₀, A₀, w₀, P...; ϵ = 1e-6, verbose = true)

  # ───────────────────────────────────────────────────────────────────
  # Constrained Support Vector Machine
  #
  # (aᵢ, yᵢ) are features/labels for point i
  # aᵢ are rows of A
  # yᵢ ∈ {-1,1}
  #
  # Primal
  # ------
  # minimize. w₀ᵀmax{0, Y₀(A₀x + ρ) - 1}
  #      s.t. w₁ᵀmax{0, Y₁(A₁x + ρ) - 1} ≦ 1
  #           ⋮
  #           wᵣᵀmax{0, Yᵣ(Aᵣx + ρ) - 1} ≦ 1
  # Dual
  # ------
  # minimize. ½‖(YA)ᵀu‖² + eᵀu + eᵀt
  #      s.t. 0 ≦ u₀ ≦ w₀
  #           0 ≦ u₁ ≦ w₁t₁
  #           ⋮
  #           0 ≦ uᵣ ≦ wᵣtᵣ
  #           uᵀe = 0
  #
  # Recover x = (YA)ᵀu
  #         ρ = 
  # ───────────────────────────────────────────────────────────────────  

  (y_,A_,w_,m_) = parseargs(y₀, A₀, w₀, P)

  k     = Int(length(P)/3)

  y     = sparse(cat(1, y_...)) 
  w     = sparse(cat(1, w_...))
  A     =        cat(1, A_...)
  wc    = sparse(cat(1, zeros(m_[1],k), w_[2:end]...))

  (m,n) = size(A)

  YA    = sparse(y.*A)
  Q     = [ YA; zeros(k,n) ]

  # ───────────────────────────────────────────────────────────────────  
  # Inequality Constraints
  # ───────────────────────────────────────────────────────────────────
  
  #      m            k                   │
  Z = [  speye(m, m)  spzeros(m,k)  ; # m │ u ≧ 0   
        -speye(m, m)  wc            ] # m │ u₀ ≦ w₀  
  z = [  zeros(m,1)       ;           #   │
        -w₀               ;
         zeros(m-m_[1],1) ] 

  U = Z[:,1:m]
  V = Z[:,m+1:end]

  # ───────────────────────────────────────────────────────────────────
  # Linear Constraints
  # ───────────────────────────────────────────────────────────────────

  G = sparse([ y' zeros(1,k) ])    
  g = spzeros(1,1)

  # ───────────────────────────────────────────────────────────────────
  # Solve
  # ┌                  ┐ ┌   ┐   ┌   ┐
  # │ QQᵀ + Z'F²Z   G' │ │ x │   │ a │
  # │ G                │ │ y │ = │ b │
  # └                  ┘ └   ┘   └   ┘
  #
  # by solving the sparse 4x4 system
  # 
  #  U = Z[1:m]      
  #  V = Z[m+1:m+k]  
  #  
  # and noting that 
  #          ┌    ┐    ┌     ┐   ┌              ┐
  #  Z'F²Z = │ U' │ F² │ U V │ = │ U'F²U  U'F²V │
  #          │ V' │    └     ┘   │ V'F²U  V'F²V │
  #          └    ┘              └              ┘
  # ┌                         ┐ ┌   ┐   ┌   ┐
  # │ U'F²U  U'F²V  (YA)'  G  │ │ x │   │ a │ size m 
  # │ V'F²U  V'F²V            │ │ r │ = │ 0 │ size k 
  # │ YA             -I       │ │ s │ = │ 0 │ size n 
  # │ G'                      │ │ y │ = │ b │ size 1 
  # └                         ┘ └   ┘   └   ┘  
  # ───────────────────────────────────────────────────────────────────
  function solve2x2_sparseinv(F)

    F² = inv(F[1]*F[1]).diag

    UᵀF²U = U'*(F².*U)
    UᵀF²V = U'*(F².*V)
    VᵀF²V = V'*(F².*V)

    Γ = [ UᵀF²U     UᵀF²V          YA            y             ;
          UᵀF²V'    VᵀF²V          spzeros(k,n)  spzeros(k,1)  ;
          YA'       spzeros(n,k)  -speye(n,n)    spzeros(n,1)  ;
          y'        spzeros(1,k)   spzeros(1,n)  spzeros(1,1)  ]

    Γ = ldltfact(Γ)

    function solve2x2(x,y) 

      xystar = Γ\[x; zeros(n); y]
      return (xystar[1:(m+k)]'', xystar[end:end]')
    
    end

  end

  QQᵀ = IntPoint.SymWoodbury(spdiagm(zeros(m+k)), Q, eye(n));

  sol = IntPoint.intpoint(QQᵀ, [ones(m,1);-1*ones(k,1)],
                          Z, z, [("R", 2*m )],
                          G, g,
                          optTol = ϵ,
                          maxRefinementSteps = 5,
                          solve2x2gen = solve2x2_sparseinv,
                          verbose = verbose)

  vsol = sol.y[1:end-k]
  xsol = A'*(y.*vsol)
  λsol = sol.y[end-k+1:end]
  ρsol = bias( y, A, cat(1, w_[1], λsol.*w_[2:end]...), xsol )

  return (xsol, ρsol, vsol, λsol)

end

# ───────────────────────────────────────────────────────────────────
# Wrapper around LIBSVM
# ───────────────────────────────────────────────────────────────────
function svm(y, A, w; ϵ = 1e-6)

  # Do training
  model  = svm_train(y, A', w, verbose = false, eps = ϵ)
  (x, ρ) = get_primal_variables(model)
  v      = abs(get_dual_variables(model))

  # pred(A :: Matrix) = 
  #   [svm_predict(model, A[i,:][:])[2][1] for i = 1:size(A,1)]
  
  # pred(A :: SparseMatrixCSC{Float64, Int64}) = 
  #   [svm_predict(model, full(A[i,:])[:])[2][1] for i = 1:size(A,1)]
  
  return (x, ρ, v)

end

function svm_intpoint(y, A, w; ϵ = 1e-6)

  # Do training
  (x, ρ, v, _) = svmc(y, A, w, verbose = false, ϵ = ϵ*10e-4)
  return (x, ρ, v)

end
# ───────────────────────────────────────────────────────────────────
# Same interface as svmc, except using LIBSVM.
# ───────────────────────────────────────────────────────────────────  
function svmcbisect(y₀, A₀, w₀, P...; ϵ = 1e-6, verbose = false)

  (y_,A_,w_,m_) = parseargs(y₀, A₀, w₀, P)

  y = cat(1, y_...)
  A = cat(1, A_...)

  xρv = Array{Any}(3)

  # Todo fix some precision issues, as well as starting interval for search
  # 0-2 works well
  function svm_coverage_weights!(t, ϵ = 2.0, τ = 10.)

    # Construct weighted SVM
    w      = [w_[1];t*w_[2]];

    # Do training
    (x, ρ, v) = svm(y,A,w,ϵ = ϵ)

    # Calculate Primal/Dual objective values
    pval   = vecdot(w, max( 1 - (y.*(A*x - ρ)[:]), 0)) + 0.5*norm(x)^2
    dval   = 0.5*norm(A'*(y.*v))^2 - sum(v)
    
    # Gradient
    (m₀,n) = size(A₀)

    gval   = vecdot(w_[2], max( 1 - (y_[2].*(A_[2]*x - ρ)) , 0))

    # Log information with each evaluation
    xρv[1] = x; xρv[2] = ρ; xρv[3] = v;

    return ((x, ρ, v), (-pval + τ*t, dval + τ*t, gval - τ))

  end

  λ = bisect( (x,ϵ) -> svm_coverage_weights!(x,ϵ,1)[2] ;
              il = 0, iu = 10000/mean(w_[2]), 
              tol = 5*mean(w_[2]) )

  return (xρv[1], xρv[2], xρv[3], λ)

end

function svmramp(y₀, A₀, w₀, P...; ϵ = 1e-6, verbose = true)

  (y,A,w,m) = parseargs(y₀, A₀, w₀, P)

  k = length(w)

  # Initilization
  δ = Array{Any}(k) 
  I = Array{Any}(k)
  RVal = ones(k)
  HVal = ones(k)

  for j = 1:k
    I[j] = y[j] .> -Inf
    δ[j] = 0
  end

  ρ = 0; λ = 0; x = zeros(n) # Initial Valuess

  R(x) = min(max(1 - x, 0), 2)
  H(x) = max(1 - x, 0)

  println()
  println("     i ┃  R₀          H₀          R₁          H₁          t")
  println("  ┎────────────────────────────────────────────────────────────────── ")

  for i = 1:10
    
    (x,ρ,v,λ) = svmc( y[1][I[1]], A[1][I[1],:], w[1][I[1]],             # Objective
                          y[2][I[2]], A[2][I[2],:], w[2][I[2]]/(1 - δ[2]),  # Constraint
                          verbose = false )                  

    for j = 1:length(y)

      z = y[j].*(A[j]*x - ρ)

      I[j] = ( z .> -1 )[:] 
      δ[j] = 2*sum(w[j][ !I[j] ]) # Points upper bounded by 1
      
      if δ[j] < 0; println("Infeasible! ", δ[j]); break; end

      RVal[j] = vecdot( w[j], R(z) )
      HVal[j] = vecdot( w[j], H(z) )

    end

    ξ() = @printf("  ┃ %2i ┃ % 06.3e  % 06.3e  % 06.3e  % 06.3e   %i\n", 
                  i, 
                  RVal[1], RVal[2], 
                  RVal[2] - 1, 
                  HVal[2] - 1, 
                  sum(I[1]) + sum(I[2])); ξ();

    if abs(RVal[2] - 1) < 1e-3
      println("  ┖────────────────────────────────────────────────────────────────── ")  
      return (x, ρ, v)
    end

  end

  println("  ┖────────────────────────────────────────────────────────────────── ")    
  return (x, ρ, v)

end