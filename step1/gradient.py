import math

f = lambda x: x * math.sin(x) # target function
nabla = lambda x: (math.sin(x) + x * math.cos(x)) # gradient function

def gradient(init, direction, eps=1e-6, alpha=0.01):
    """
    @param1 : Iteration starting point
    @param2 : Gradient direction (+1:Up, -1:Down)
    @param3 : Convergence condition
    @param4 : Iteration step
    """
    x0 = init
    while True:
        x = x0 - direction * alpha * nabla(x0) # iterative operator
        if abs(x - x0) < eps: # set convergence condition
            break
        if x < -6.0: # set x lower limit
            break
        if x > 8.0: # set x upper limit
            break
        x0 = x
    return x

def get_extremum(point):
    """
    @param1 : Iteration starting point
    @return1: max value of f(x)
    @return2: min value of f(x)
    """
    x1 = gradient(point, 1)
    x2 = gradient(point, -1)

    y1 = f(x1)
    y2 = f(x2)
    if y1 > y2:
        return (y1, y2)
    else:
        return (y2, y1)

if __name__ == '__main__':

    list = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    maximum = minimum = f(4.5)
    for pt in list:
        (max_val, min_val) = get_extremum(pt)
        if maximum < max_val:
            maximum = max_val
        if minimum > min_val:
            minimum = min_val
    print ("f = x * sin(x) , x = [ -6.0, 8.0 ]")
    print ("f_max= %.6f, f_min = %.6f" %(maximum, minimum))
