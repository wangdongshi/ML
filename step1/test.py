def grad_dec(eps=1e-8, delta=0.001):
    """
    :param eps: 函数值误差
    :param delta: 迭代步长
    """
    x0 = 3.0
    f = lambda a: a * a - 3.0 * a + 5.0
    while True:
        x = x0 - delta * (2.0 * x0 - 3.0)
        if abs(x - x0) < eps: # 指定横坐标收敛跳出循环
            break
        x0 = x
    print(x, f(x))


if __name__ == '__main__':
    grad_dec()

