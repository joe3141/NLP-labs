import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk import pos_tag, sent_tokenize, word_tokenize, FreqDist
from nltk.corpus import words

from collections import Counter

import enchant
import pyphen

import time

corpus = []
labels = []


for root, dirs, files in os.walk("lingspam_public/bare/"):
    for file in files:
    	with open(os.path.join(root, file), "r") as f:
    		corpus.append(f.read().strip())
    		if file[0] == 's': # Spam?
    			labels.append(1)
    		else:
    			labels.append(0)

# Shuffles and splits
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=4)



# BOW
vectorizer = CountVectorizer(stop_words="english")
# Vectorize, append labels, transform to dataframe
train = pd.DataFrame(np.hstack((vectorizer.fit_transform(X_train).toarray(), np.array(y_train)[:, None])))
test  = pd.DataFrame(np.hstack((vectorizer.transform(X_test).toarray(), np.array(y_test)[:, None])))

train.to_csv("train.csv", header=False, index=False)
test.to_csv("test.csv", header=False, index=False)

spam_list = []
with open("spam_word_list.txt", "r") as f:
	spam_list = [word.strip().lower() for word in f.readlines() if word != "\n"]

d = enchant.Dict("en_US")
pyphen.language_fallback('nl_NL_variant1')
dic = pyphen.Pyphen(lang='en_GB')


def extract_features(doc):
	doc = doc.lower()
	res = []
	tokens = word_tokenize(doc)
	sents = sent_tokenize(doc)
	# Number of sentences
	res.append(len(sents))

	# Number of verbs
	tags = pos_tag(tokens)
	counts = Counter(token[1] for token in tags)
	res.append(counts["VB"])

	# Number of words that are found in the spam list
	spam_list_no = 0
	for w in spam_list:
		if w in doc:
			spam_list_no += 1

	res.append(spam_list_no)

	# Number of spelling mistakes. Currently, not sensitive to other languages.
	# Number of words that contain both numeric and alphabetical chars,
	typos = 0 
	nums = 0
	
	# Number of words with more than 3 syllables
	three_syl_no = 0
	# Avg. number of syllables,
	avg_syl_word = 0
	word_no = 0

	# Sum of TF-ISF, Term frequence-Inverse sentence frequency
	tf_isf = 0.0
	f_terms = FreqDist(tokens)

	for token in tokens:
		# Checks if this token is an English word
		# if token in words.words(): # It might be a proper word with no typos from a different language
		if not d.check(token):
			typos +=1

		syl_res = dic.inserted(token)
		syls_no = len(syl_res.split("-"))

		if syls_no > 3:
			three_syl_no += 1

		word_no += 1
		avg_syl_word += syls_no

		# Not just numbers and contains at least one digit
		if not (token.isdigit()) and any(c.isdigit() for c in token):
			nums += 1

		tf = float(f_terms[token])
		isf = 0.0
		for s in sents:
			if token in s:
				isf += 1.0

		if isf > 0.0:
			isf = (float(len(sents))) / isf
		else:
			isf = 0.0
		tf = 1.0 + np.log(tf)
		tf_isf += tf * isf


	avg_syl_word /= word_no
	res.extend((typos, nums, three_syl_no, avg_syl_word, tf_isf))

	return res

vv = TfidfVectorizer(stop_words='english', sublinear_tf=True)
train_tf_idf = np.sum(vv.fit_transform(X_train).toarray(), axis=1)
test_tf_idf = np.sum(vv.transform(X_test).toarray(), axis=1)

start = time.time()

train = np.hstack((np.array([extract_features(doc) for doc in X_train]), train_tf_idf[:, None], \
	np.array(y_train)[:, None]))

test = np.hstack((np.array([extract_features(doc) for doc in X_test]), test_tf_idf[:, None], \
	np.array(y_test)[:, None]))

print('It took', time.time()-start, 'seconds.')

np.savetxt("train_f.csv", train, delimiter=",")
np.savetxt("test_f.csv", test, delimiter=",")
