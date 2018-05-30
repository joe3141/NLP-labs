import os
from nltk import pos_tag, word_tokenize, help
from nltk.corpus import brown
from nltk import FreqDist
from collections import defaultdict

# print(pos_tag(word_tokenize('The man ate the apple')))
# help.brown_tagset('DT')
# help.brown_tagset()

# word_tags = set()

# for root, dirs, files in os.walk("single-docs/"):
# 	for file in files[:50]:
# 			f = open(os.path.join(root, file), 'r')
# 			words = pos_tag(word_tokenize(f.read()))
# 			word_tags = word_tags.union(words)

# print([i[0] for i in word_tags if i[1][:2] == 'NN'])


all_tagged = brown.tagged_sents()
train = all_tagged[:50000]
test = all_tagged[50000:]

tags = defaultdict(lambda: FreqDist())

for i in train:
	for word, pos in i:
		tags[word][pos]+=1

# print(tags)

# print(tags['play'].most_common())


unigram_tagger = defaultdict(str)
for word in tags:
	unigram_tagger[word] = tags[word].most_common(1)[0][0]

print(unigram_tagger)
