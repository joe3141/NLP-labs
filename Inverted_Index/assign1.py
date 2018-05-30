import os
import sys
from nltk import word_tokenize
from nltk.util import ngrams


def create_index(n):
	bigram_index = {}

	for root, dirs, files in os.walk("single-docs"):
		for file in files[:n]:
			with open(os.path.join(root, file), "r") as f:
				tokens = word_tokenize(f.read().lower())
				bigrams = ngrams(tokens, 2)

				for bigram in bigrams:
					if not bigram in bigram_index:
						bigram_index[bigram] = []
					bigram_index[bigram].append(file)

	return bigram_index


def main(num_of_files, first_word, second_word):
	index = create_index(num_of_files)
	query = (first_word.lower(), second_word.lower())
	# print(index)
	if query in index:
		result = index[query]
		
		for f in result:
			print(f)
	else:
		print("Not found.")


if __name__ == '__main__':
	main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
