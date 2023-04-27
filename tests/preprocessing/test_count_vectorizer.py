import pytest
import numpy as np

from inventronet.preprocessing import CountVectorizer


@pytest.fixture
def documents():
    return [
        "This is a simple document.",
        "This document is about programming.",
        "Another document is related to Python."
    ]


@pytest.fixture
def expected_vocabulary():
    return {
        'this': 0,
        'is': 1,
        'a': 2,
        'simple': 3,
        'document': 4,
        'about': 5,
        'programming': 6,
        'another': 7,
        'related': 8,
        'to': 9,
        'python': 10
    }


def test_count_vectorizer_fit_transform_vocabulary(documents, expected_vocabulary):
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(documents)
    assert vectorizer.vocabulary_ == expected_vocabulary


def test_count_vectorizer_fit_transform_output(documents, expected_vocabulary):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)

    expected_output = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    ])

    assert np.array_equal(X, expected_output)


def test_count_vectorizer_transform_shape(documents, expected_vocabulary):
    vectorizer = CountVectorizer()
    vectorizer.fit(documents)

    new_documents = [
        "A new document about Python.",
        "This is a simple test."
    ]

    X_new = vectorizer.transform(new_documents)
    assert X_new.shape == (2, len(expected_vocabulary))


def test_count_vectorizer_lowercasing_shape(documents):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    assert X.shape == (3, len(vectorizer.vocabulary_))


def test_count_vectorizer_no_lowercasing_shape(documents):
    vectorizer_no_lowercase = CountVectorizer(lowercase=False)
    X_no_lowercase = vectorizer_no_lowercase.fit_transform(documents)
    assert X_no_lowercase.shape == (3, len(vectorizer_no_lowercase.vocabulary_))


def test_count_vectorizer_no_lowercasing_vocabulary_size():
    documents = [
        "This is a Simple document.",
        "This DOCUMENT is about programming.",
        "Another document is related to Python."
    ]

    vectorizer = CountVectorizer()
    vectorizer.fit_transform(documents)

    vectorizer_no_lowercase = CountVectorizer(lowercase=False)
    vectorizer_no_lowercase.fit_transform(documents)

    assert len(vectorizer_no_lowercase.vocabulary_) > len(vectorizer.vocabulary_)

