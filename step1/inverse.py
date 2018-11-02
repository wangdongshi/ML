import random

def inverse(n):
	"""
	@param1 : number of input array
	"""
	P = []
	nsum = n * (1 + n) / 2
	for i in range(0, n):
		isum = i * (1 + (i + 1)) / 2
		P.append(isum/nsum)
	
	#print(P)
	x = n
	k = 1
	rnd = random.uniform(0, 1)
	while k < n:
		if rnd < P[k-1] :
			x = k
			break
		k = k + 1

	print('random=%f, x=%s'%(rnd, x))
	
#if __name__ == '__main__':
for i in range (0, 15):
	inverse(10)

