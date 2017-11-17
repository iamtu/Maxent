from scipy.optimize import fmin_l_bfgs_b

def f_grad(x):
    return [(x-10)*(x-10) + 1, 2*(x-10)]

def call_me(x):
    print 'current x:', x
    return
if __name__ == '__main__':
    x = -8
    y, f_min, dic = fmin_l_bfgs_b(f_grad, x, iprint = 0, callback = call_me)
    print 'y = ', y
    print 'f_min = ', f_min