from .base import BaseTextSplitter
from .sentence import SentenceTextSplitter

TextSplitter = SentenceTextSplitter

__all__ = ["BaseTextSplitter", "SentenceTextSplitter", "TextSplitter"]
