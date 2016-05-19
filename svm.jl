using IntPoint
using LIBSVM
using LIBLINEAR
using DataFrames

include("bisect.jl")
include("misc.jl")
#include("slicingbundle.jl")

toplot = false

function duality_gap(y, A, w, pre, v)

    # Calculate Primal/Dual objective values
    # x      = A'*(y.*v)
    # pval   = vecdot(w, max( 1 - (y.*pred(A)[:]), 0)) + 0.5*norm(x)^2
    # dval   = 0.5*norm(A'*(y.*v))^2 - sum(v)
    SV   = v .!= 0
    z    = pre(A)
    α    = vecdot(z,y.*v)
    pval = vecdot(w, max( 1 - y.*z, 0)) + 0.5*α
    dval = 0.5*α - sum(v)

    return pval, dval

end

function diagdot(A)
  
  n = size(A,1)
  c = zeros(n)
  for i = 1:size(A,1)
    c[i] = vecdot(sub(A,i,:), sub(A,i,:))
  end
  return c

end

function dmat(A,B)

  n = size(A,1)
  m = size(B,1)

  println(n," ",m," ",size(A),size(B))

  diagdot(A)*ones(1,m) + ones(n,1)*diagdot(B)' - 2*A*B'

end

function rbfmat(A,B; degree = 2, coef0 = 1, γ = 1.) 

  exp(-dmat(A,B)*γ)

end

function polymat(A,B; degree = 2, coef0 = 1, γ = 1.)

  (γ*A*B' + coef0).^degree

end

function tanhmat(A,B; degree = 2, coef0 = 1, γ = 1.)

  tanh(γ*A*B' + coef0)

end

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
  #      s.t. w₁ᵀmax{0, Y₁(A₁x + ρ) - 1} ≦ η₁
  #           ⋮
  #           wᵣᵀmax{0, Yᵣ(Aᵣx + ρ) - 1} ≦ ηᵣ
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
  if k != 0
    wc    = sparse([zeros(m_[1],k); cat([1 2], w_[2:end]...)])
  else
    wc = sparse(zeros(m_[1],k))
  end

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

  return (A -> A*xsol - ρsol, vsol, λsol)

end

# ───────────────────────────────────────────────────────────────────
# Wrapper around various SVM solvers
# ───────────────────────────────────────────────────────────────────
function svm_libsvm(y,A,w; 
  ϵ = 1e-6,
  kernel::Int32 = Int32(0),
  γ = 1.,
  coef0 = 1.,
  degree = 2,
  verbose = false)

  # Indicies get messed up when there are 0 weights. So remove them
  # before passing them into LIBSVM

  Inz = !(w .== 0)

  # Do training
  model  = svm_train(y[Inz], A[Inz,:]', w[Inz], 
                     verbose = verbose, 
                     eps = ϵ,
                     kernel_type = kernel,
                     gamma = γ,
                     coef0 = coef0,
                     degree = degree)

  # Get variables we need
  v          = spzeros(size(w,1),1)
  v₀         = abs(get_dual_variables(model))
  v[Inz,:]   = v₀

  mdl = unsafe_load(model.ptr)
  ρ   = unsafe_load(mdl.rho)
  SV  = v .!= 0

  pred = nothing

  if kernel == 0
    (x, ρ) = get_primal_variables(model)
    pred = A -> (A*x - ρ)
  end

  if kernel == 1
    pred = D -> (polymat(D,A[SV,:];
                 degree=degree,coef0=coef0,γ=γ)*((v.*y)[SV]) - ρ)[:]
  end

  if kernel == 2
    pred = D -> (rbfmat(D,A[SV,:]; 
                 degree=degree,coef0=coef0,γ=γ)*((v.*y)[SV]) - ρ)[:]
  end

  if kernel == 3
    pred = D -> (tanhmat(D,A[SV,:]; 
                 degree=degree,coef0=coef0,γ=γ)*((v.*y)[SV]) - ρ)[:]
  end

  return (pred, v)

end

function svm_intpoint(y, A, w; ϵ = 1e-6, verbose = verbose)

  # Do training
  (pred, v₀) = svmc(y, A, w, verbose = verbose, ϵ = 1e-8)

  return (pred, v₀)

end

function svm_liblinear(y,A,w; ϵ = 1e-5, verbose = true)

  A = A'

  # Sort data, remove 0 weights
  P  = sortperm(y, rev = true)
  P  = P[w[P] .!= 0]

  yP = y[P]
  AP = A[:,P]
  wP = w[P]

  model = linear_train(yP, AP, wP; verbose=verbose, solver_type=Cint(3), eps = ϵ, bias = 1)
  vP = sparsevec(model.SVI + 1, model.SV, size(y,1))
  v = spzeros(size(y,1),size(y,2))
  
  v[P] = vP

  ρ = model.w[end]
  x = model.w[1:end-1]

  return(A -> (A*x + ρ), v)

end

function svm(y,A,w; 
  ϵ = 1e-6,
  kernel::Int32 = Int32(0),
  γ = 1.,
  coef0 = 1.,
  degree = 2,
  verbose = true,
  solver = :liblinear)

  if solver == :libsvm

    return svm_libsvm(y,A,w; 
      ϵ = ϵ, 
      kernel = kernel,
      γ = γ, 
      coef0 = coef0, 
      degree = degree,
      verbose = verbose )
  
  end


  if solver == :liblinear

    if kernel != Int32(0)
      throw("Nonlinear kernels not supported for liblinear")
    end
    return svm_liblinear(y,A,w; ϵ = ϵ, verbose = verbose)

  end
  
  if solver == :svm_intpoint

    if kernel != Int32(0)
      throw("Nonlinear kernels not supported for interior point")
    end
    return svm_intpoint(y,A,w; ϵ = ϵ, verbose = verbose)
    
  end

end

# ───────────────────────────────────────────────────────────────────
# Same interface as svmc, except using LIBSVM.
# ───────────────────────────────────────────────────────────────────
function svmc_bisect(y₀, A₀, w₀, P...;
  tol = 0.1,
  maxiters = 20,
  verbose = false,
  kernel = Int32(0),
  γ = 1.,
  coef0 = 1.,
  degree = 1,
  solver = :libsvm)

  (y_,A_,w_,m_) = parseargs(y₀, A₀, w₀, P)

  y = cat(1, y_...)
  A = cat(1, A_...)

  pred_v = Array{Any}(3)

  # Todo fix some precision issues, as well as starting interval for search
  # 0-2 works well
  function svmw!(t, ϵ = 2.0, τ = 10.)

    # Construct weighted SVM
    w      = [w_[1];t*w_[2]];

    pre = x -> x

    # Do training
    (pre, v) = svm(y,A,w,
                   ϵ = ϵ,
                   kernel = kernel,
                   γ = γ,
                   coef0 = coef0,
                   degree = degree,
                   verbose = true,
                   solver = solver)

    (pval, dval) = duality_gap(y, A, w, pre, v)

    # Gradient
    (m₀,n) = size(A₀)

    gval   = vecdot(w_[2], max( 1 - (y_[2].*pre(A_[2])) , 0))

    # Log information with each evaluation
    pred_v[1] = pre; pred_v[2] = v;

    return (-pval + τ*t, dval + τ*t, gval - τ)

  end

  λ = bisect( (x,ϵ) -> svmw!(x,ϵ,1) ;
              il = 1e-5, iu = 10/mean(w_[2][w_[2] .!= 0]),
              tol = tol*mean(w_[2][w_[2] .!= 0]) ,
              verbose = verbose,
              maxiters = maxiters)

  pred = pred_v[1]
  v    = pred_v[2]

  return (pred, v, λ)

end

# ───────────────────────────────────────────────────────────────────
# Ramp version of LIBSVM.
# ───────────────────────────────────────────────────────────────────

function svmramp(y₀, A₀, w₀, P...; ϵ = 1e-6, verbose = true, 
  maxIters        = 100,
  optTol          = 1e-10,
  kernel::Int32   = Int32(0),
  γ::Real         = 1.,
  coef0::Real     = 1.,
  degree::Integer = 2,
  solver          = :liblinear)

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

  n = size(A₀,1)
  ρ = 0; λ = 0; x = zeros(n) # Initial Valuess

  R(x)  = min(max(1 - x, 0), 2)
  #∇R(x) = if (x > 1 || x < -1); { return 0 } else { return -1; } end;
 
  H(x) = max(1 - x, 0)

  if verbose == true
    println()
    println("     i ┃  R₀          H₀          R₁          H₁          t")
    println("  ┎────────────────────────────────────────────────────────────────── ")
  end

  pred = nothing
  v = nothing
  λ = nothing

  wᵢ = deepcopy(w)

  for i = 1:8

    (pred,v,λ) = svmc_bisect( y[1], A[1], wᵢ[1],  # Objective
                             y[2], A[2], wᵢ[2],  # Constraint
                             verbose = false ,
                             kernel = kernel,
                             γ = γ,
                             coef0 = coef0,
                             degree = 2 ,
                             solver = solver)

    # (pred,v,λ) = svmc( y[1], A[1], wᵢ[1],  # Objective
    #                    y[2], A[2], wᵢ[2], verbose = false, ϵ = 2e-5)

    for j = 1:length(y)

      z = y[j].*pred(A[j])

      I[j] = ( z .> -1 )[:]
      δ[j] = 2*sum(w[j][ !I[j] ]) # Points upper bounded by constant

      if j > 1
        wᵢ[j][:] = (w[j]/(1 - δ[j]))[:]
      else
        wᵢ[j][:] = w[j][:]
      end

      wᵢ[j][!I[j]] = 0;

      if δ[j] < 0; println("Infeasible! ", δ[j]); break; end

      RVal[j] = vecdot( w[j], R(z) )
      HVal[j] = vecdot( w[j], H(z) )

    end

    if verbose == true
      ξ() = @printf("  ┃ %2i ┃ % 06.3e  % 06.3e  % 06.3e  % 06.3e   %i\n",
                    i,
                    RVal[1], RVal[2],
                    RVal[2] - 1,
                    HVal[2] - 1,
                    sum(I[1]) + sum(I[2])); ξ();
    end

    if abs(RVal[2] - 1) < optTol*mean(w[2])

      if verbose == true
        println("  ┖────────────────────────────────────────────────────────────────── ")
      end

      return (pred, v, λ)

    end

  end

  if verbose == true
    println("  ┖────────────────────────────────────────────────────────────────── ")
  end

  return (pred, v, λ)

end

function svm_oracle(y₀, A₀, w₀, P...; solver_params...)

  (y_,A_,w_,m_) = parseargs(y₀, A₀, w₀, P)

  y = cat(1, y_...)
  A = cat(1, A_...)

  n = length(y_) - 1
  τ = ones(n)

  function Ω(t, ϵ = 1e-6)

    # Construct weighted SVM
    z = [1;t]; w = cat(1,[z[i]*w_[i] for i = 1:length(w_)]...)

    (pre, v) = svm(y,A,w;
                   ϵ = ϵ,
                   solver_params...)

    (pval, dval) = duality_gap(y, A, w, pre, v)

    gval = zeros(0)
    
    for i = 2:(n+1)
      gi = vecdot(w_[i], max( 1 - (y_[i].*pre(A_[i])) , 0))
      push!(gval, gi)
    end
    
    return (-pval + vecdot(τ,t), dval + vecdot(τ,t), -gval + τ, pre, v)

  end

  return Ω

end

function svmc_epicut(y₀, A₀, w₀, P...; solver_params...)

  (y_,A_,w_,m_) = parseargs(y₀, A₀, w₀, P)
  nc            = length(w_)
  ORACLE        = svm_oracle(y₀, A₀, w₀, P...; solver_params...)

  η = Float64[mean(w_[i][w_[i] .!= 0]) for i = 2:nc]

  # Generate Initial Polyhedra
  P₀ = box( length(η)+1   , 
           [ 0 ; 0.0001 ] , 
           [ 0 ; (3./η) ] )

  P₀.A = P₀.A[2:end,:]; P₀.b = P₀.b[2:end] # Remove original lower bound
  (l,u,g,_,_) = ORACLE(η*1.5, 1.0);        # Generate lower bound from oracle  

  # Lower bound from oracle
  b = vecdot(η/2,g) - l
  push!(P₀, [-1 g[:]'], b + abs(b)) # Make a more conserverative lowe bound

  λ             = sbundle((x,ϵ) -> ORACLE(x,ϵ)[1:3], P₀, max_iters = 10)
  (_,_,_,pre,v) = ORACLE(λ)

  return (pre, v, λ)

end

# ───────────────────────────────────────────────────────────────────
# Variation on vanilla SVM which allows the margin to be shifted
# for individual data components
# ───────────────────────────────────────────────────────────────────

function svmshift(y, A, w, s, ϵ = 1e-4, verbose = true)

  # ───────────────────────────────────────────────────────────────────
  # Constrained Support Vector Machine
  #
  # (aᵢ, yᵢ) are features/labels for point i
  # aᵢ are rows of A
  # yᵢ ∈ {-1,1}
  #
  # Primal
  # ------
  # minimize. wᵀmax{0, Y₀(A₀x + ρ) - s}
  # Dual
  # ------
  # minimize. ½‖(YA)ᵀu‖² + zᵀu 
  #      s.t. 0 ≦ u ≦ w
  #
  # Recover x = (YA)ᵀu
  # ───────────────────────────────────────────────────────────────────

  (m,n) = size(A)

  A  = [ones(size(A,1),1) A]; n = n + 1
  YA = sparse(y.*A)
  Q  = YA

  # ───────────────────────────────────────────────────────────────────
  # Inequality Constraints
  # ───────────────────────────────────────────────────────────────────

  Z = [  speye(m, m) ; # m │ u ≧ 0
        -speye(m, m) ] # m │ u₀ ≦ w₀
  
  z = [  zeros(m,1)    ;           #   │
        -w             ;]

  # ───────────────────────────────────────────────────────────────────
  # Solve
  # QQᵀ + Z'F²Z
  # ───────────────────────────────────────────────────────────────────
  QQᵀ = IntPoint.SymWoodbury(Diag(zeros(m)), full(Q), eye(n));

  function solve2x2_sparseinv(F)

    v = inv(F[1]*F[1]).diag
    D = Diag(v[1:m] + v[m+1:end])
    invHD = IntPoint.SymWoodburyMatrices.liftFactor(QQᵀ + D);
    return (rhs, _ )  -> (invHD(rhs), zeros(0,1));

  end

  sol = IntPoint.intpoint(QQᵀ, s'',
                          Z  , z  , [("R", 2*m )];
                          optTol = ϵ,
                          maxRefinementSteps = 4,
                          solve2x2gen = solve2x2_sparseinv,                          
                          verbose = verbose)

  vsol = sol.y
  xsol = A'*(y.*vsol)

  return (A -> [ones(size(A,1),1) A]*xsol, vsol)

end

function loadmodel(y₀, A₀, w₀, P...; solver = :liblinear, v = nothing, λ = nothing)

  if solver == :liblinear
    
    (y_,A_,w_,m_) = parseargs(y₀, A₀, w₀, P)

    y = cat(1, y_...)
    A = cat(1, A_...)
    
    x = A'*(y.*v)
    ρ = sum(y.*v)

    return (A -> A*x + ρ, v)
  end


end

function test1()

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
  @time (x1,ρ1,v1) = svm_libsvm(y, A, w);

end

function test2()

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
  w           = ones(2*d)

  (pred,v) = svmcbisect(y, A, w, ones(size(y)), A, w/500, verbose = true)

end
