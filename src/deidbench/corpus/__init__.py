"""Synthetic annotated corpus for measuring PHI detection.

Reproducible from a seed, synthetic throughout, and annotated by location class
so that results say *where* a detector fails rather than only how often.
"""

from .annotation import Annotation, AnnotatedMessage, Corpus, LocationClass
from .generate import GenerationOptions, generate_corpus
from .people import Person, make_person

__all__ = [
    "AnnotatedMessage",
    "Annotation",
    "Corpus",
    "GenerationOptions",
    "LocationClass",
    "Person",
    "generate_corpus",
    "make_person",
]
