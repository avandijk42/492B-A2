from __future__ import division
import random
from math import log
from collections import defaultdict

def getCharacterHistogram(fileName):
	f = open(fileName)
	# defaults to small value instead of 0 to prevent errors in cross-entropy calculation
	histo = defaultdict(lambda:10**-6)
	for char in f.read():
		histo[char] += 1
	return dict(histo)

def getCharacterDistribution(histo):
	total = sum(histo.values())
	for key in histo.keys():
		histo[key] /= total
	return histo

def getTrainTest(tweets):
	trainSize = 2*len(tweets) // 3
	trainSet = random.sample(tweets, trainSize)
	testSet = [ t for t in tweets if t not in trainSet ]
	return trainSet, testSet

def drawNSamplesFromDistro(n, distro):
	samples = []
	for _ in range(n):
		samples.append(drawWeightedSampleFromDistro(distro))
	return samples

def drawWeightedSampleFromDistro(distro):
	ordered_distro = sorted(distro.items(), key=lambda x: x[1])
	cumsum = 0.0
	ksi = random.uniform(0,1)
	for char, prob in ordered_distro:
		cumsum += prob
		if ksi < cumsum:
			break
	return char

def crossEntropy(p, X):
	N = len(X)
	negNormLog = lambda x: (-1/N)*log(p[x])
	return sum([negNormLog(x_i) for x_i in X])

def perplexity(p, X):
	return 2 ** crossEntropy(p,X)

