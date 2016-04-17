# ───────────────────────────────────────────────────────────────────
# Utility for doing plots.
# ───────────────────────────────────────────────────────────────────

function plotline(x₁, x₂, x, l, g) 

  plot([x₁; x₂], [(l +(x - x₁)*(g)); (l + (x - x₂)*(g))], 
       "k",linewidth = 0.5)
  plot([x],[l],"k-")
  #plot([x],[u],"k.")

end

# ───────────────────────────────────────────────────────────────────
# Find  the minimum of f, where is a 1D convex function.
# (l,u,g) = f(x) where l + tg ≦ f(x) ≦ u ∀t
# ───────────────────────────────────────────────────────────────────

function bisect(f; tol = 0.001, il = 0, iu = 2, verbose = false, ε₀ = 0.001)

  if toplot
    clf()
    L = zeros(0); U = zeros(0)
    R = linspace(il,iu,50)
    for t = R
      (l,u,g) = f(t, 0.01)
      push!(L,l); push!(U,u)
    end

    plot(R,L,"r",linewidth = 2); hold(true); plot(R,U)
    ax = axis()
    axis(ax)
    plot(R,L,"r",linewidth = 2)
  end

  LB = zeros(0); UB = zeros(0); G = zeros(0); X = zeros(0)

  lowerbound(x) = maximum(LB + G.*(X - x))

  function lowerboundinv(fx)

    # x[i] is the x location where line i intersects fx
    x = (-fx + LB + G.*X)./G 

    Iv = [-Inf, Inf]    
    for xi = x
      (lb,i) = findmax(LB + G.*(X - xi)) # sorta like lowerbound
      if abs(lb - fx) < 1e-7; 
        G[i] <= 0 ? Iv[2] = xi : Iv[1] = xi
      end
    end

   return Iv

  end

  if verbose
    println()
    println("     i ┃    fₓ-f*          ‖x-x*‖         gₓ            r    ")
    println("  ┎─────────────────────────────────────────────────────────── ")
  end

  # Interval [0,2] containing the solution
  (l,u,g) = f(il, ε₀);
  if toplot; plotline(0,2,il,l,g); end;
  push!(LB, l); push!(UB, u); push!(G, g); push!(X, il)

  # this indicates the constraint is inactive, i.e. the lagrange multiplier
  # is 0. No bisection is needed. TODO see if this can be integrated 
  # elegantly into the for loop.
  if g < 0
    if verbose
      println("  ┖─────────────────────────────────────────────────────────── ")  
    end
    return 0
  end

  (l,u,g) = f(iu, ε₀)
  if toplot; plotline(0,2,iu,l,g); end;
  push!(LB, l); push!(UB, u); push!(G, g); push!(X, iu)

  for i = 1:10
    
    ilp = il; iup = iu;
    (il, iu) = lowerboundinv(minimum(UB))
    if !isfinite(il) || !isfinite(iu); 
      throw("Solution is either infeasible, unbounded, or does not lie "*
            "in bound [$ilp , $iup]"); 
    end

    x = (il + iu)/2.
    ϵ = min(abs(g), 1)/1000

    j = 0
    for j = 1:3
      # If solution is not accuratae, restart the whole thing.
      (l,u,g) = f(x, ϵ)
      if abs((l - u)/(g*(il - iu))) < 0.3; break; end
      ϵ = ϵ/100
    end

    push!(LB, l); push!(UB, u); push!(G, g); push!(X, x)
    
    if verbose
      e() = @printf("  ┃ %2i ┃   % 06.3e     % 06.3e     % 06.3e     %1i\n", 
                    i, u-l, iu-il, g, j); e();
    end

    if toplot; plotline(0,2,x,l,g); end;

    if abs(g) < tol; 
      if verbose
        println("  ┖─────────────────────────────────────────────────────────── ")
      end
      return x; 
    end

  end

  if verbose
    println("  ┖─────────────────────────────────────────────────────────── ")
  end

end