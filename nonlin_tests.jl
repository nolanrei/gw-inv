

BIG = 1e5
nw = length(waves)
nz = length(col1.U)
J = zeros(2nz,nw)
H = zeros(nw,nw)
gi= zeros(2nz)
lambda = 1e-6
p = (L81_exp_grad!,waves,col1,momdep_tst1,lambda,BIG,gi,J)
x0 = x#-5*ones(length(waves))

grad_tmp = zeros(2nz)
G = zeros(nw)
G2 = zeros(nw)

tiny = 1e-5*ref_flux

## L81_grad test
#=
gi .= 0.0
J .= 0.0
flux1 = log(0.5*ref_flux)
L81_exp_grad!(waves[22],col1,flux1,gi,grad_tmp)
grad_truth = zeros(2nz)
grad_tmp2 = zeros(2nz)
L81_exp_grad!(waves[22],col1,flux1+tiny,grad_truth,grad_tmp2)
grad_truth .-= gi
grad_truth ./= tiny
for i = 1:2nz
    println(i, "\t", grad_truth[i], "\t", grad_tmp[i])
end
=#

## test Jacobian

## nl_opt gradient test
gi .= 0.0
J .= 0.0
G .= 0.0
G2 .= 0.0
bsl = nl_obj(x0,p)
nl_obj_grad!(G,x0,p)
gi .= 0.0
J .= 0.0
x2 = x0 .+ 0.0
for i = 1:nw
    x2[i] += tiny
    G2[i] = (nl_obj(x2,p) - bsl)/tiny
    x2[i] = x0[i]
end
for i = 1:nw
    println(wavespds[i], "\t", G2[i], "\t", G[i])
end
