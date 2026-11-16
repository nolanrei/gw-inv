import numpy as np

class Wave:
    def __init__(self, c, k, flux, direction, source_lvl):
        self.c = c
        self.k = k
        self.uw_flux = flux #np.abs(k*c*A)
        self.dir = direction
        self.srclvl = source_lvl


class Profile:
    def __init__(self, U, V, rho, N):
        self.U = U
        self.V = V
        self.rho = rho
        self.N = N

def L81(c, k, flux, dir, src_lvl, col):
    """
    Corresponds to "loop from bottom to top; if flux > b, deposit the difference as momentum and reduce flux to b"

    ONLY works if directions of waves are all the same
    """
    vel = col.U*dir[0] + col.V*dir[1]       # mean flow speed in direction of wave
    c = np.array(c)
    if len(c.shape) == 0:
        # then c was a scalar going in, and we want it a (1,) vector
        c = np.expand_dims(c,0)

    flux = np.array(flux)
    if len(flux.shape) == 0:
        flux = flux*np.ones(c.shape)

    # breaking criterion -- maximum wave flux at which the wave is stable
    # This version is signed! We need the sign to catch places where c = u in between datapoints
    # (and therefore (c-u) changes sign). We abs() it later on.
    c = np.expand_dims(c,0)
    vel = np.expand_dims(vel,1)
    nk = c.shape[1]
    nz = c.shape[0]
    tmp = (c-vel).T        # vectors are row matrices under the hood in numpy, so we need to match
    b = col.rho/col.rho[src_lvl] * (k*(tmp)**3)/(2*col.N)            # b is now nz x nk
    b = b.T
    sign_change = np.argmin(np.sign(b)*np.sign(b[src_lvl,:]), axis=0)  # find sign changes down columns

    # Since Python sucks we can't do this the natural way, in a loop
    # Using the cumulative minimum isolates the parts where the pipe is getting narrower
    # min_dep finds the first index where momflux > b
    # first prevents looking at critical layers below the wave's source
    b = np.minimum.accumulate(np.abs(b), axis=0)
    for ic in range(nk):
        if sign_change[ic] > 0:
            # if a critical layer happens, all further momentum transport is blocked
            b[sign_change[ic]:,ic] = 0
    db = b[1:,:]-b[:-1,:]
    min_dep = np.argmax(b < np.abs(flux.T), axis=0)-1    # the -1 is to start before the spike, not after
    bool1 = (min_dep < 0)                        # if there was no breaking, argmax returns 0 (which would deposit momentum near surface)
    min_dep[bool1] = b.shape[0]                  # this index should be out of bounds, avoiding spurious momdep
    first = np.maximum(min_dep, src_lvl)

    momdep = np.zeros(b.shape)
    for ic in range(nk):
        momdep[first[ic]+1:,ic] = np.minimum(-db[first[ic]:,ic],np.abs(flux[ic])-b[first[ic]+1:,ic])*np.sign(flux[ic])

    return np.sum(momdep,axis=1)

def AD99(c, k, flux, dir, src_lvl, col):
    """
    Implementation of the Alexander-Dunkerton 1999 gravity wave parameterization
    Corresponds to looping from bottom to top, and when flux > capacity, deposit all momentum.
    """
    vel = col.U*dir[0] + col.V*dir[1]       # mean flow speed in direction of wave

    # breaking criterion -- maximum wave flux at which the wave is stable
    # This version is signed! We need the sign to catch places where c = u in between datapoints
    # (and therefore (c-u) changes sign). We abs() it later on.
    b = col.rho/col.rho[src_lvl] * (k*(c-vel)**3)/(2*col.N)
    sign_change = np.argmin(np.sign(b)*np.sign(b[src_lvl]))

    # Since Python sucks we can't do this the natural way, in a loop
    # Using the cumulative minimum isolates the parts where the pipe is getting narrower
    # min_dep finds the first index where momflux > b
    # first prevents looking at critical layers below the wave's source
    b = np.minimum.accumulate(np.abs(b))
    if sign_change > 0:
        # if a critical layer happens, all further momentum transport is blocked
        b[sign_change:] = 0
    db = b[1:]-b[:-1]
    min_dep = np.argmax(b < flux)-1      # the -1 is to start before the spike, not after
    if min_dep < 0:
        # argmax/argmin return 0 if the condition is never attained. We don't want this.
        min_dep = db.shape[0]
    first = np.maximum(min_dep, src_lvl)

    # this is the difference from L81 -- all momentum is deposited at breaking point, not f-b of it
    momdep = np.zeros(db.shape)
    momdep[first] = flux

    return momdep

def lin_invert(propag, col, momdep, wavespds, src_lvl, ref_flux=1e-3, reg=1e-3):
    """
    To a close approximation, gravity wave propagation and breaking is linear:
    parameterizations are completely noninteracting between wavenumbers (except for MSGWaM), and
    variations in amplitude are slightly nonlinear when critical layers are not full-occlusion.
    Thus, we can approximate the wave propagation problem
                      g(X, S) = D
    by the linear problem
                      G_X*S = D.
    G_X has dimensions (nz x nk), where nz is #(pts in the vertical) and nk is # wavespeeds.
    If these are small enough we can construct G_X explicitly as G_X[:,i] = g(X,e_i)
    In principle, the solve could be done by Krylov space methods if forming the matrix is impractical.
    """
    nz = momdep.shape[0]
    nk = wavespds.shape[0]

    # basics for forming the columns
    k = 2*np.pi/1000      # 1km wavelength
    direction = np.array((1.0,0.0))
    #ref_flux = 1e-3    # rescaling to not pick up too much detail. builds G_X*ref_flux, so remember to divide by it at the end
    # need to assume a source level too --> src_lvl

    # construct the matrix
    GX = np.zeros((nz,nk))
    for i in range(nk):
        # the i:i+1 keeps the last dimension unsqueezed (possibly unnecessary now)
        GX[:,i] = propag(wavespds[i], k, ref_flux, direction, src_lvl,col)
    GX /= ref_flux    # rescaling. Remember?

    # solve!
    U,S,Vh = np.linalg.svd(GX,False)
    # cut off the singular values smaller than eps*max(M,N)
    rcond = np.maximum(nk,nz)*np.finfo(np.float64).eps
    reliable = (S >= rcond)
    S = S[reliable]
    U = U[:,reliable]
    Vh = Vh[reliable,:]
    # ridge regression
    S += reg
    spectrum = Vh.T@((U.T@momdep)/S)

    return spectrum, GX

def lin_invert2d(propag, col, momdep, wavespds, src_lvl, ref_flux=1e-3, reg=1e-3):
    """
    To a close approximation, gravity wave propagation and breaking is linear:
    parameterizations are completely noninteracting between wavenumbers (except for MSGWaM), and
    variations in amplitude are slightly nonlinear when critical layers are not full-occlusion.
    Thus, we can approximate the wave propagation problem
                      g(X, S) = D
    by the linear problem
                      G_X*S = D.
    G_X has dimensions (nz x nk), where nz is #(pts in the vertical) and nk is # wavespeeds.
    If these are small enough we can construct G_X explicitly as G_X[:,i] = g(X,e_i)
    In principle, the solve could be done by Krylov space methods if forming the matrix is impractical.
    """
    nz = momdep.shape[0]//2
    nk = wavespds.shape[0]

    # basics for forming the columns
    k = 2*np.pi/1000      # 1km wavelength
    # need to assume a source level too --> src_lvl
    #ref_flux = 1e-3    # rescaling to not pick up too much detail. builds G_X*ref_flux, so remember to divide by it at the end

    # Adding more directions (in case it helps)
    num_directions = 4
    isqrt2 = np.sqrt(2)
    directions_master = np.array([[1.0, 0.0], [isqrt2, isqrt2], [isqrt2, -isqrt2], [0.0, 1.0]])

    # construct the matrix
    GX = np.zeros((2*nz,num_directions*nk))
    for d in range(num_directions):
        direction = directions_master[d, :]
        for i in range(nk):
            GX[:nz,i+d*nk] = direction[0]*propag(wavespds[i], k, ref_flux, direction, src_lvl, col)
            GX[nz:,i+d*nk] = direction[1]*propag(wavespds[i], k, ref_flux, direction, src_lvl, col)
    GX /= ref_flux    # rescaling. Remember?

    # solve!
    U,S,Vh = np.linalg.svd(GX,False)
    # cut off the singular values smaller than eps*max(M,N)
    rcond = np.maximum(nk,nz)*np.finfo(np.float64).eps
    reliable = (S >= rcond)
    S = S[reliable]
    U = U[:,reliable]
    Vh = Vh[reliable,:]
    # ridge regression
    S += reg
    spectrum = Vh.T@((U.T@momdep)/S)

    return spectrum, GX, directions_master