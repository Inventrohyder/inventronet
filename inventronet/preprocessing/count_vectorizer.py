import re
from collections import Counter, defaultdict
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
        return doc.lower() if self.lowercase else doc

    def _tokenize(self, doc: str) -> List[str]:
        return [m.group(0) for m in re.finditer(self.token_pattern, doc)]

    def fit(self, documents: List[str]) -> None:
        word_counts = Counter(word for doc in documents for word in self._tokenize(self._preprocess(doc)))

        if self.max_features is not None:
            self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(self.max_features))}
        else:
            self.vocabulary_ = {word: idx for idx, word in enumerate(word_counts.keys())}

    def transform(self, documents: List[str]) -> csr_matrix:
        if self.vocabulary_ is None:
            raise RuntimeError("You need to call fit() before calling transform().")

        indptr = [0]
        indices = []
        data = []
        vocabulary = defaultdict(lambda: None, self.vocabulary_)

        for doc in documents:
            preprocessed_doc = self._preprocess(doc)
            tokenized_doc = self._tokenize(preprocessed_doc)
            word_counts = Counter(tokenized_doc)

            for word, count in word_counts.items():
                index = vocabulary[word]
                if index is not None:
                    indices.append(index)
                    data.append(count)

            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=int, shape=(len(documents), len(self.vocabulary_)))

    def fit_transform(self, documents: List[str]) -> csr_matrix:
        self.fit(documents)
        return self.transform(documents)
