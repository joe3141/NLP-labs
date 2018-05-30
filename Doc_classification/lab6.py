from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernouliNB

cats = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
newsgroups = fetch_20newsgroups(subset ='all', categories=cats)
train = fetch_20newsgroups(subset='train', categories=cats)
test = fetch_20newsgroups(subset='test', categories=cats)

# v = CountVectorizer()
# transformed = v.fit_transform(train.data)

# # print(transformed)

# clf = MultinomialNB(alpha=1)
# clf.fit(transformed, train.target)

test_t = v.fit(test)
