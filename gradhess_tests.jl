# gradient test for the softmin propagator

f0 = nl_obj(x,p)
nl_obj_grad!(mem[3],x,p)

df = zeros(nw)
ep = 1e-6
xp = zeros(nw)

for i = 1:nw
    xp .= x
    xp[i] += ep
    f1 = nl_obj(xp,p)
    df[i] = (f1-f0)/ep
end

plot(wavespds,mem[3],xlabel="wavespeed (m/s)",ylabel="gradient (Pa^2/Pa)",label="nl_obj_grad!")
plot!(wavespds,df,label="finite diff")
savefig("testfig2.png")

# Hessian test

# 1. Use a better step size for 2nd derivatives
ep = 1e-3
iep2 = 1.0 / (ep * ep)
hessd = zeros(nw, nw)

# Run the analytical Hessian to compare later
nl_obj_hess_noreg!(hessnoreg, x, p)

# 2. Precompute 1D perturbations to avoid O(N^2) redundant calls
f_plus = zeros(nw)
f_minus = zeros(nw)
xp = copy(x)

for i in 1:nw
    xp[i] = x[i] + ep
    f_plus[i] = nl_obj(xp, p)
    
    xp[i] = x[i] - ep
    f_minus[i] = nl_obj(xp, p)
    
    xp[i] = x[i] # Reset xp
end

# 3. Build the numerical Hessian
for j in 1:nw
    # Diagonal case (i=j)
    hessd[j,j] = (f_plus[j] - 2*f0 + f_minus[j]) * iep2
    
    # Off-diagonal case (i < j)
    for i in 1:j-1
        # Evaluate the two mixed 2D perturbations
        xp[i] = x[i] + ep
        xp[j] = x[j] + ep
        f_pp = nl_obj(xp, p)
        
        xp[i] = x[i] - ep
        xp[j] = x[j] - ep
        f_mm = nl_obj(xp, p)
        
        # Reset xp
        xp[i] = x[i]
        xp[j] = x[j]
        
        # Apply  exact stencil using the cached 1D evaluations
        tmp = (2*f0 + f_pp + f_mm - f_plus[i] - f_minus[i] - f_plus[j] - f_minus[j]) * 0.5 * iep2
        
        hessd[i,j] = tmp
        hessd[j,i] = tmp
    end
end