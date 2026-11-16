include("omp.jl")

function test_single_wave_recovery()
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
    wav = c_to_wave(10*randn(),[0.0 1.0],kw,ref_flux,src)
    os = OMPStruct(col, L81_AD_wrapper!, wav; max_nw=2)
    
    # 2. Define a true wave [theta, c, flux]
    true_x = [0.5, 20.0, 0.005]
    os.fluxvec .= 0.0
    os.prop_AD!(os.fluxvec, true_x, os.wav, os.col) # Ground truth signal
    
    # 3. Attempt to recover using FB sharpen logic
    x_guess = [0.2, 10.0, 0.001] # Initial guess
    ws = FullHessianOMPWorkspace(50, 1)
    fgh = make_full_hessian_fgh(os, ws, 1e-8, os.sc)
    
    # Run optimizer with explicit bounds
    # NewtonTrustRegion doesn't take bounds, use ConjugateGradient or LBFGS
    # if you want to support constraints.
    res = Optim.optimize(Optim.only_fgh!(fgh), x_guess, LBFGS())
    
    println("Recovered: ", res.minimizer)
    println("True:      ", true_x)
end