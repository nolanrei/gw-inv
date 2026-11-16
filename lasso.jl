using Lasso

## objective is to minimize 
#        ||Ax-b||^2 + \lambda |x|_1,
# i.e., L1 regularization for linear systems (sometimes called LASSO)
# Misleadingly, I will be using an interior point method and not the 
# LASSO algorithm, mostly because I like it better.


