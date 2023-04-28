import re
from collections import Counter
from typing import List, Dict, Union, Optional

from scipy.sparse import csr_matrix


class CountVectorizer:
    def __init__(
            self,
            lowercase: bool = True,
            token_pattern: str = r'\b\w+\b',
            max_features: Optional[int] = None,
    ):
        self.lowercase: bool = lowercase
        self.token_pattern: str = token_pattern
        self.vocabulary_: Union[Dict[str, int], None] = None
        self.max_features = max_features

    def _preprocess(self, doc: str) -> str:
        if self.lowercase:
            doc = doc.lower()
        return doc

    def _tokenize(self, doc: str) -> List[str]:
        return re.findall(self.token_pattern, doc)

    def fit(self, documents: List[str]) -> None:
        word_counts = Counter()
        for doc in documents:
            doc = self._preprocess(doc)
            tokens = self._tokenize(doc)
            word_counts.update(tokens)

        if self.max_features is not None:
            word_counts = word_counts.most_common(self.max_features)
            self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(word_counts)}
        else:
            self.vocabulary_ = {word: idx for idx, word in enumerate(word_counts.keys())}

    def transform(self, documents: List[str]) -> csr_matrix:
        if self.vocabulary_ is None:
            raise RuntimeError("You need to call fit() before calling transform().")

        indptr = [0]
        indices = []
        data = []
        for doc in documents:
            doc = self._preprocess(doc)
            tokens = self._tokenize(doc)
            word_counts = Counter(tokens)
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    index = self.vocabulary_.get(word)
                    indices.append(index)
                    data.append(count)
            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=int)

    def fit_transform(self, documents: List[str]) -> csr_matrix:
        self.fit(documents)
        return self.transform(documents)
