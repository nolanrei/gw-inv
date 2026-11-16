using LinearAlgebra
using FFTW
using Integrals

function mknu(k,nu,N2)
    # vertical wvnum m as a fn. of horz. wvnum k and freq. nu
    return -sign(nu)*k*sqrt(N2/(nu*nu) - 1.0)
end

function Bknu(k,nu,N2,Q0,Q_nu,G_k,h)
    mkv = mknu(k,nu,N2)
    mh = mkv*h
    return (k*k)/(nu*nu)*G_k*Q0*Q_nu*(pi/mh)*sin(mh)/(mkv*mkv-pi*pi*h*h)
end

function Fknu(k,nu,params)
    N2,Q0,Q,G,h,rho0,C,u,c = params
    # Q needs to be a function of real variable nu, even though it represents a Fourier decomposition
    # C = 1/sqrt(2pi) 1/(L*tau), where L and tau are length and time intervals (e.g. 1000km and 1s)
    B = Bknu(k,nu,N2,Q0,Q(nu),G(k),h)
    mt = sign(nu)*sqrt(N2/(nu*nu)-1.0)
    return C*rho0*mt*B*B
end

function integrand(nu,params)
    N2,Q0,Q,G_k,h,rho0,C,u,c = params
    t1 = nu*nu/N2
    t2 = nu*(1-t1)
    k = (c-u)/t2
    dkdnu = -k/t2*(1-3*t1)
    return Fknu(k,nu,params)*sqrt(1.0+dkdnu*dkdnu)
end

function Fc(c,params)
    # Flux as a function of wavespeed c in the Beres 2004 model
    params[9] = c    # we need to pass wavespeed as a parameter since we're integrating over nu
    domain = (0.0, sqrt(params[1]))   # (0,N)
    prob = IntegralProblem(integrand,domain,params)
    sol = solve(prob, QuadGKJL())
    return sol.u
end

