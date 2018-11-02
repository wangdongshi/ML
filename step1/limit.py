import math

f = lambda x: x * math.sin(x) # target function
nabla = lambda x: math.sin(x) + x * math.cos(x)		# gradient function
f1 = lambda x: math.sin(x) + x * math.cos(x) 		# first derivative
f2 = lambda x: 2 * math.cos(x) - x * math.sin(x)	# second derivative

def newton(init, direction, eps=1e-6):
    """
    @param1 : Iteration starting point
    @param2 : Iteration direction (+1:Up, -1:Down)
    @param3 : Convergence condition
	@return1: Convergence point
	@return2: Convergence count
    """
    count = 0
    x0 = init
    while True:
        count += 1
        x = x0 - direction * f1(x0) / f2(x0) # iterative operator
        if abs(x - x0) < eps: # set convergence condition
            break
        if x < -6.0: # set x lower limit
            break
        if x > 8.0:  # set x upper limit
            break
        x0 = x
    return (x, count)

def gradient(init, direction, eps=1e-6, alpha=0.01):
    """
    @param1 : Iteration starting point
    @param2 : Gradient direction (+1:Up, -1:Down)
    @param3 : Convergence condition
    @param4 : Iteration step
	@return1: Convergence point
	@return2: Convergence count
    """
    count = 0
    x0 = init
    while True:
        count += 1
        x = x0 - direction * alpha * nabla(x0) # iterative operator
        if abs(x - x0) < eps: # set convergence condition
            break
        if x < -6.0: # set x lower limit
            break
        if x > 8.0: # set x upper limit
            break
        x0 = x
    return (x, count)

if __name__ == '__main__':

    list = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    print ("f = x * sin(x) , x = [ -6.0, 8.0 ]")
    
    # Newton
    sum_step = 0
    min_x = f(0.0)
    for pt in list:
	    (lower, count) = newton(pt, 1)
	    if min_x > lower: min_x = lower
	    sum_step += count
    min_y = f(min_x)
    print ("----- Newton -----")
    print ("x_min = %.6f, y_max= %.6f, step = %d" %(min_x, min_y, sum_step))
    
    # Gradient (alpha=0.01)
    sum_step = 0
    min_x = f(0.0)
    for pt in list:
	    (lower, count) = gradient(pt, 1, alpha = 0.01)
	    if min_x > lower: min_x = lower
	    sum_step += count
    min_y = f(min_x)
    print ("----- Gradient (alpha = 0.01) -----")
    print ("x_min = %.6f, y_max= %.6f, step = %d" %(min_x, min_y, sum_step))
    
    # Gradient (alpha=0.1)
    sum_step = 0
    min_x = f(0.0)
    for pt in list:
	    (lower, count) = gradient(pt, 1, alpha = 0.1)
	    if min_x > lower: min_x = lower
	    sum_step += count
    min_y = f(min_x)
    print ("----- Gradient (alpha = 0.1) -----")
    print ("x_min = %.6f, y_max= %.6f, step = %d" %(min_x, min_y, sum_step))
    
    # Gradient (alpha=0.3)
    sum_step = 0
    min_x = f(0.0)
    for pt in list:
	    (lower, count) = gradient(pt, 1, alpha = 0.3)
	    if min_x > lower: min_x = lower
	    sum_step += count
    min_y = f(min_x)
    print ("----- Gradient (alpha = 0.3) -----")
    print ("x_min = %.6f, y_max= %.6f, step = %d" %(min_x, min_y, sum_step))
    
