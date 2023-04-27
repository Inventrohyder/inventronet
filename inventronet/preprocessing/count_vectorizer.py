import re
from collections import Counter

import numpy as np


class CountVectorizer:
    def __init__(self, lowercase=True, token_pattern=r'\b\w+\b'):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.vocabulary_ = None

    def fit(self, documents):
        word_counts = Counter()
        for doc in documents:
            if self.lowercase:
                doc = doc.lower()
            tokens = re.findall(self.token_pattern, doc)
            word_counts.update(tokens)

        self.vocabulary_ = {word: idx for idx, word in enumerate(word_counts.keys())}

    def transform(self, documents):
        if self.vocabulary_ is None:
            raise RuntimeError("You need to call fit() before calling transform().")

        matrix = []
        for doc in documents:
            if self.lowercase:
                doc = doc.lower()
            tokens = re.findall(self.token_pattern, doc)
            row = [tokens.count(word) for word in self.vocabulary_.keys()]
            matrix.append(row)

        return np.array(matrix)

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
