function ppp(x, K, N, cgh)
    a = -1/(K*N*N)
    b = 1/K
    c = cgh
    x1 = (a*x*x+b)
    px = x1*x+c
    pp = x1 + x*(2*a*x)
    return px, pp
end

function cgz(cgh, K)
    N = 0.02
    om = -cgh*K
    for i = 1:6
        px, pp = ppp(om, K, N, cgh)
        om -= px/pp
    end
    m = sqrt(abs(K*K*(N*N-om*om)/om/om))
    cgz = abs(m*om/(K*K+m*m))
    return cgz
end