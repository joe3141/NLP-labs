import os
import numpy as np
from collections import defaultdict
from operator import itemgetter # Faster than using a lambda
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

# def ingest_docs(path):
	
# 	res = []

# 	for root, dirs, files in os.walk(path):
# 		for file in files:
# 			with open(os.path.join(root, file), 'r') as f:
# 				res.append((f.read(), int(file)))


# 	return [d[0] for d in sorted(res, key=itemgetter(1))] 


def ingest_docs(path):
	lines = [l.strip() for l in open(path, 'r').readlines()]
	docs = []
	acc = ""
	ingest = False

	for l in lines:
		if l.startswith(".W"):
			ingest = True
			acc = ""
			continue

		if ingest:
			if l.startswith(".I"):
				ingest = False
				docs.append(acc)
			else:
				acc += l
	docs.append(acc)

	return docs

def apk(ref, res, k):

	if (len(ref) == 0) or (len(res) == 0):
		return 0.0

	if len(res) > k:
		res = res[:k]

	score = 0.0
	num_hits = 0.0

	for i, p in enumerate(res):
		if p in ref: # Supposing there are no duplicates in res.
			num_hits += 1.0
			score += num_hits / (i+1.0) # Precision @ i

	return score / min(len(ref), k)


docs = ingest_docs('cran/original/cran.all.1400')
queries = ingest_docs('cran/original/cran.qry')
crans = [l.strip().split() for l in open('cran/cranqrel', 'r').readlines()]

vv = TfidfVectorizer(stop_words='english', sublinear_tf=True)
doc_tfidf = vv.fit_transform(docs)
query_tfidf = vv.transform(queries)

# print(np.isfinite(query_tfidf.toarray()).all())
# print(np.isfinite(doc_tfidf.toarray()).all())

with np.errstate(invalid='ignore', divide='ignore'):
	sims = (1 - cdist(doc_tfidf.toarray(), query_tfidf.toarray(), metric='cosine')).T # T because I want sims for every q
# print(sims.shape)
query_idx = [(np.argsort(sim)[::-1] + 1) for sim in sims]

judgs = defaultdict(list)
for j in crans:
	judgs[int(j[0])].append(int(j[1]))

MAP = 0.0

for i in range(225):
	ap = apk(judgs[i+1], query_idx[i], len(query_idx[i]))
	MAP += ap

MAP /= 225.0

print(MAP)