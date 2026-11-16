include("omp.jl")

# Have data already loaded

function probfilter(ds::DimStack; nsamp=1000, src=10, top=55, f0=5e-4, coords=nothing, min_deposition=0.05)
    # NOTE: src and top are in KILOMETERS, not indices

    if isnothing(coords)
        cmax=80
        nc = (2cmax+1)^2
        coords = zeros(2,nc)
        idx = 0
        for cx=-cmax:cmax, cy=-cmax:cmax
            idx += 1
            coords[1,idx] = atan(cy,cx)
            coords[2,idx] = hypot(cx,cy)
        end
    end
    
    sz = size(ds[:U])
    nz = sz[3]
    kw = 2pi/1e4
    wav_dummy = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

    max_nw=20
    ws = FullHessianOMPWorkspace(nz, max_nw)

    N = zeros(nz)
    rho = zeros(nz)
    z = zeros(nz)
    dz = zeros(nz)   #it's entirely unused
    
    col_dummy = ColumnProfile(ds[:U][1,1,:,1], ds[:V][1,1,:,1], N, rho, nz, dz)
    os = OMPStruct(col_dummy, L81_AD_wrapper!, wav_dummy; max_nw=max_nw)

    nx = 3
    x = zeros(nx)
    flux = zeros(2nz)

    xs,ys,ts = rand(1:sz[1],nsamp), rand(1:sz[2],nsamp), rand(1:sz[4],nsamp)
    nc = size(coords,2)
    p = zeros(nc)

    for i = 1:nsamp
        xi,yi,ti = xs[i], ys[i], ts[i]
    
        # Initialize column profile
        sample_existing_dims!(os.col.N, ds, :N, xi, yi, ti)
        sample_existing_dims!(os.col.rho, ds, :rho, xi, yi, ti)
        sample_existing_dims!(z, ds, :z, xi, yi, ti)
        srci = bsrch(src,z)
        topi = bsrch(top,z)
        
        os.wav.src = srci
        os.col.U .= ds[:U][xi,yi,:,ti]
        os.col.V .= ds[:V][xi,yi,:,ti]

        for j = 1:nc
            # Initialize x and clean out flux
            x[1] = coords[1,j]
            x[2] = coords[2,j]
            x[3] = f0
            fill!(flux,0)
            #println(x)
            fwd!(flux, x, nx, os)
            #println(flux)
            # Condition!
            bfx = hypot(flux[srci], flux[srci+nz])
            tfx = hypot(flux[topi], flux[topi+nz])
            if abs(1 - tfx/bfx) > min_deposition
                p[j] += 1
            end
        end
    end
    p ./= nsamp
    return p
end

pf = probfilter(wrf_ds;f0=5e-4)#; nsamp=1, coords=[0.0 pi/2; 10.0 10.0])
heatmap(-80:80,-80:80,reshape(pf,(161,161)),ratio=1);savefig("pf10_01.png")