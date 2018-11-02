import random

def shuffle(n):
	"""
	@param1 : number of input array
	"""
	poker = []
	for i in range(0, n): poker.append(i)
	
	k = n
	while k > 1:
		i = int(random.uniform(0,1) * k) + 1
		poker[i-1],poker[k-1] = poker[k-1],poker[i-1]
		k = k - 1

	print(poker)
	
if __name__ == '__main__':

	for i in range (0, 10):
		shuffle(10)

