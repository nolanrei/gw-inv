epses = range(-0.01,stop=0.01,length=100)
ne = length(epses)
fs = zeros(ne)
fapxs = zeros(ne)
f0 = nl_obj(x,p)

function nl_obj_hess_noreg!(H,x,p)
    # computes Hessian of nl_obj w/ respect to x in H
    # J must be size nz x nw
    propag!,propag_2der!,waves,col,b,lambda,BIG,Gi,J,der2,w,wndw,wndw_flag = p
    @assert typeof(propag_2der!) <: Function "p[2] must be 2nd deriv function"
    # need to clean up Gi and J storage arrays
    eps = 1e-12
    Gi .= 0.0
    J .= 0.0
    der2 .= 0.0
    nw = length(waves)
    for i = 1:nw
        # additively builds up Gi into g(S;X) while building up jacobian
        propag!(waves[i],col,x[i],Gi,view(J,:,i))
        # fills in second derivative
        propag_2der!(waves[i],col,x[i],view(der2,:,i))
    end
    Gi .-= b
    if wndw_flag
        # window Gi to only look at a subsection of the column
        Gi .*= wndw
        for i = 1:nw
            J[1:nz,i] .*= wndw
            J[nz+1:2nz,i] .*= wndw
            der[1:nz,i] .*= wndw
            der[nz+1:2nz,i] .*= wndw
        end
    end
    H .= 2*BIG*(J'*J)
    for i = 1:nw
        # second derivative correction
        correction = dot(Gi,view(der2,:,i))
        # cuts off negative corrections since they can lead to 
        # the Newton step not being a descent direction
        corr_throttle = 1.0#(correction > 0.0 ? 1.0 : 0.0) 
        H[i,i] += 2*BIG*corr_throttle*correction
        # +eps is undocumented regularization preventing small xi from crashing inversion
        H[i,i] += BIG*lambda*w[i]*exp(x[i]) + eps
    end
    return nothing
end

hessnoreg = zeros(nw,nw)
nl_obj_hess_noreg!(hessnoreg,x,p)
fapxnoreg = zeros(ne)

for i = 1:ne
    foo = epses[i]*mem[1]
    fs[i] = nl_obj(x.+foo,p)
    fapxs[i] = f0 + mem[3]'*foo + foo'*mem[4]*foo
    fapxnoreg[i] = f0 + mem[3]'*foo + foo'*hessnoreg*foo
end

plot(epses,(fs),label="true nl_obj")
plot!(epses,(fapxs),label="quadratic appx to nl_obj")
plot!(epses,(fapxnoreg),label="quadratic appx, no regularization")
savefig("Figures/WACCM/Reconstructions/testfig.png")