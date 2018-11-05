import sys
import numpy as np

def markov(p1, p2, p3):
    # print initial distribution
    PI0 = np.array([p1, p2, p3])
    print('initial distribution {:.3f} {:.3f} {:.3f}' .format(PI0[0], PI0[1], PI0[2]))

    # markov matrix
    P = np.array([[.65, .28, .07],
                  [.15, .67, .18],
                  [.12, .36, .52]])
    x = PI0
    PIn = [x]
    for i in range(10):
        print('{:.3f} {:.3f} {:.3f}' .format(x[0], x[1], x[2]))
        x = np.dot(x, P)
        PIn.append(x)
    print ('\n')


#iitial distribution
markov(0.21, 0.68, 0.11)
markov(0.38, 0.40, 0.22)
markov(0.17, 0.14, 0.69)
import sys
import numpy as np

def markov(p1, p2, p3):
    # print initial distribution
    PI0 = np.array([p1, p2, p3])
    print('initial distribution {:.3f} {:.3f} {:.3f}' .format(PI0[0], PI0[1], PI0[2]))
    
    # markov matrix
    P = np.array([[.65, .28, .07],
                  [.15, .67, .18],
                  [.12, .36, .52]])
    x = PI0
    PIn = [x]
    for i in range(10):
        print('{:.3f} {:.3f} {:.3f}' .format(x[0], x[1], x[2]))
        x = np.dot(x, P)
        PIn.append(x)
    print ('\n')


#iitial distribution
markov(0.21, 0.68, 0.11)
markov(0.38, 0.40, 0.22)
markov(0.17, 0.14, 0.69)
