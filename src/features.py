import re
from difflib import SequenceMatcher

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def lexical_features(reference, answer):
    ref_tokens = tokenize(reference)
    ans_tokens = tokenize(answer)

    ref_set = set(ref_tokens)
    ans_set = set(ans_tokens)

    overlap = len(ref_set & ans_set) / max(len(ref_set), 1)
    jaccard = len(ref_set & ans_set) / max(
        len(ref_set | ans_set), 1
    )

    length_ratio = len(ans_tokens) / max(len(ref_tokens), 1)
    word_count_difference = abs(
        len(ans_tokens) - len(ref_tokens)
    )

    char_similarity = SequenceMatcher(
        None,
        str(reference).lower(),
        str(answer).lower(),
    ).ratio()

    return [
        overlap,
        jaccard,
        length_ratio,
        word_count_difference,
        char_similarity,
    ]


def fit_vectorizer(train_df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )

    texts = (
        train_df["instructor_answer"].astype(str).tolist()
        + train_df["student_answer"].astype(str).tolist()
    )

    vectorizer.fit(texts)
    return vectorizer


def extract_features(df, vectorizer):
    references = df["instructor_answer"].astype(str).tolist()
    answers = df["student_answer"].astype(str).tolist()

    ref_matrix = vectorizer.transform(references)
    ans_matrix = vectorizer.transform(answers)

    tfidf_sim = np.asarray(
        ref_matrix.multiply(ans_matrix).sum(axis=1)
    ).ravel()

    extra_features = [
        lexical_features(ref, ans)
        for ref, ans in zip(references, answers)
    ]

    extra_features = np.asarray(extra_features, dtype=float)

    # Six interpretable features in total.
    X = np.column_stack([tfidf_sim, extra_features])

    return X


FEATURE_NAMES = [
    "tfidf_cosine",
    "reference_word_overlap",
    "jaccard_similarity",
    "answer_length_ratio",
    "word_count_difference",
    "character_similarity",
]