# Calibration of finite difference Hessian calculation

function sillyfoo(x)
    return dot(x,x)
end

function sillyhess(x)
    n = length(x)
    H = zeros(n,n)
    for i = 1:n
        H[i,i] = 2.0
    end
    return H
end

##############################
n = 5
xfoo = -14*ones(n)

# 1. Use a better step size for 2nd derivatives
ep = 1e-3 
iep2 = 1.0 / (ep * ep)
H_fd = zeros(n, n)

# Run the analytical Hessian to compare later
H_an = sillyhess(xfoo)

# 2. Precompute 1D perturbations to avoid O(N^2) redundant calls
f_plus = zeros(n)
f_minus = zeros(n)
xp = copy(xfoo)

f0 = sillyfoo(xp)

for i in 1:n
    xp[i] = xfoo[i] + ep
    f_plus[i] = sillyfoo(xp)
    
    xp[i] = xfoo[i] - ep
    f_minus[i] = sillyfoo(xp)
    
    xp[i] = xfoo[i] # Reset xp
end

# 3. Build the numerical Hessian
for j in 1:n
    # Diagonal case (i=j)
    H_fd[j,j] = (f_plus[j] - 2*f0 + f_minus[j]) * iep2
    
    # Off-diagonal case (i < j)
    for i in 1:j-1
        # Evaluate the two mixed 2D perturbations
        xp[i] = x[i] + ep
        xp[j] = x[j] + ep
        f_pp = sillyfoo(xp)
        
        xp[i] = x[i] - ep
        xp[j] = x[j] - ep
        f_mm = sillyfoo(xp)
        
        # Reset xp
        xp[i] = x[i]
        xp[j] = x[j]
        
        # Apply  exact stencil using the cached 1D evaluations
        tmp = (2*f0 + f_pp + f_mm - f_plus[i] - f_minus[i] - f_plus[j] - f_minus[j]) * 0.5 * iep2
        println((2*f0 + f_pp + f_mm - f_plus[i] - f_minus[i] - f_plus[j] - f_minus[j]) * 0.5 * iep2)
        
        H_fd[i,j] = tmp
        H_fd[j,i] = tmp
    end
end