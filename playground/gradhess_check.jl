include("omp.jl")

# 1. Setup column
sz = size(U)
xi = 94
yi = 7
t = 21
zb = 16    # 67km   (24    # 45km)
ze = 50    # 10km
nz = ze-zb+1
dz = Z[xi,yi,ze:-1:zb,t] .- Z[xi,yi,ze+1:-1:zb+1,t]
col = ColumnProfile(U[xi,yi,ze:-1:zb,t], V[xi,yi,ze:-1:zb,t],
    N[xi,yi,ze:-1:zb,t], rho[xi,yi,ze:-1:zb,t], ze-zb+1, dz)
kw = 2pi/5e4
src = 1
ref_flux = 0.5e-3
wav = c_to_wave(10*randn(),[0.0 1.0],kw,ref_flux,src)
os = OMPStruct(col, L81_AD_wrapper!, wav; max_nw=2)

# 2. Define a true wave [theta, c, flux]
true_x = [0.5, 20.0, 0.005]
os.fluxvec .= 0.0
os.prop_AD!(os.fluxvec, true_x, os.wav, os.col) # Ground truth signal

# 3. Attempt to recover using FB sharpen logic
x_guess = [0.2, 10.0, 0.001] # Initial guess
ws = FullHessianOMPWorkspace(nz, 1)
reg = 1e-8
fgh = make_full_hessian_fgh(os, ws, reg, os.sc)

# 4. Check gradient and hessian
nx = length(x_guess)
nw = div(nx,3)
g = zeros(nx)
h = zeros(nx,nx)
x_g_sc = zeros(nx)
x_to_y!(x_g_sc,x_guess,os.sc,nw)
f = fgh(0.0,g,h,x_g_sc)    # value of f in fgh doesn't matter

# forward differentiation
cost_target = y_vars -> begin
    x_phys = zeros(eltype(y_vars), nx) 
    y_to_x!(x_phys, y_vars, os.sc, nw)
    return cost(x_phys, nx, reg, os)
end
exact_grad = ForwardDiff.gradient(cost_target, x_g_sc)
exact_hess = ForwardDiff.hessian(cost_target, x_g_sc)