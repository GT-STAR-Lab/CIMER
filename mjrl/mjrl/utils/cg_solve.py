import numpy as np

def cg_solve(f_Ax, b, x_0=None, cg_iters=10, residual_tol=1e-10): 
    # objective: solving Ax = b
    # we have: x, and Ab(obtained from HVP, Hessian-vector product)
    # the maximum iter number is suggesed by TRPO paper as 10 
    # b,r,p -> dapg gradient
    x = np.zeros_like(b) #if x_0 is None else x_0   x_0 -> initial guess of the solution (in this work, it is set to be zero)
    r = b.copy() #if x_0 is None else b-f_Ax(x_0), r_0 is set to be b(vpg)
    p = r.copy() # an intermidiate parameter, it start from the vpg direction

    rdotr = r.dot(r)  # g^Tg: a scalar

    for i in range(cg_iters):  # gradient ascent for some steps (10 as default)
        z = f_Ax(p)  # input vector p will be used in eval() function of build_Hvp_eval  
        # z -> F*dk (FIM*vpg)
        # use the new p vector to compute z 
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)  
        mu = newrdotr / rdotr
        p = r + mu * p  
        rdotr = newrdotr
        if rdotr < residual_tol:  # if rdotr is sufficiently small, then we can exit loop
            break
    return x
