from __future__ import division
import random
from math import log
from collections import defaultdict
import json


## All deliverables for this assignment included in this one file for convenience
## 

# START PART 1
###

def loadParsedTweets():
	data = json.load(open('./part2.txt'))
	lines = []
	for d in data:
		line = ''
		try:
			tokens = d['tweet_parsed']
			for word in tokens:
				line = line + word + ' '
			line = line[:-1]
			lines.append(line)
		except:
			pass
	return lines

def getCharacterHistogram(allTweetsString):
	# defaults to small value instead of 0 to prevent errors in cross-entropy calculation
	histo = defaultdict(lambda:10**-6)
	for char in allTweetsString:
		histo[char] = int(histo[char] + 1)
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

###
# END PART 1

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

########################
def part2():
	lines = loadParsedTweets()
	train, test = getTrainTest(lines)
	trainDistro = getCharacterDistribution(getCharacterHistogram(''.join(train)))

	print ''.join(drawNSamplesFromDistro(140, trainDistro))
	return trainDistro, ''.join(test)
	'''
	PART 2 NOTES:

	The sampled tweets here do not look like a well-formed language. 
	This is in part because of the way in which we extracted information from the source text: the unigram
	character model does not capture enough information from the source material to reproduce
	meaningful language. In the process of extracting our probabilistic features, we are losing lots of 
	information from our source text, like what characters often appear together, or in what order. If the dimensionality of
	our feature space were greater (for instance, if we trained on a probabilistic unigram token model) 
	we would capture more information, and generate tweets with more source-language-like information.

	Furthermore, we should not treat the selection of each character as an independent event, which we 
	do in this case. In comparison to Shannon's experiment, if we asked English speakers to finish spelling
	a word, the speaker would select characters with a certain probability that is *conditioned* on the prior
	characters in the sequence. Thus, to make a better probabilistic character model, we would want to weigh
	our selection by the prior characters in the word.
	'''

def part3(trainDistro, test):
	print crossEntropy(trainDistro, test)
	print perplexity(trainDistro, test)

	'''
	PART 3 NOTES:

	The units of the cross entropy are negative log-base-2 probability, or bits, for our test samples. 
	The units of perplexity are intuitively the average of how many different outcomes might follow any other outcome. 
	Our model produces a cross-entropy of about 3.0, which is means this is a stronger model than 
	Shannon's uniform zeroth order model. The reason for our stronger model is that Shannon's model is a uniform
	model that weighs each character equally, and our model weighs characters by their relative frequencies,
	which provides more insight into what should be the outcome of an experiment.  
	'''

if __name__ == "__main__":
	trainDistro, test = part2()
	part3(trainDistro, test)