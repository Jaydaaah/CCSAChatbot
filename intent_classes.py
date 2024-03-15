from nltk_utils import lang, NLP_Util
from typing import Callable, Self
import numpy as np
import random

class Tag:
    tags_list: list["Tag"] = []
    def __init__(self, tag: str, response: Callable[[], str] | str | list[str], *args: str) -> None:
        # properties
        self.text = tag
        self.response = response
        self.patterns: list[Pattern] = []
        
        # create Pattern instances
        for arg in args:
            pat = Pattern(arg)
            self.patterns.append(pat)
            
        self.tags_list.append(self)

    def __int__(self) -> int:
        return self.tags_list.index(self)
    
    def __str__(self) -> str:
        return self.text
    
    def execute_response(self):
        if type(self.response) is str:
            return self.response
        elif type(self.response) is list:
            return random.choice(self.response)
        return self.response()
            
    @staticmethod
    def get_tag(tag: str | int) -> "Tag":
        if type(tag) is int:
            return Tag.tags_list[tag]
        elif type(tag) is str:
            for t in Tag.tags_list:
                if t.text == tag:
                    return t
        raise NotImplementedError(f"Error type({type(tag)}) with value {tag}")
    
    @staticmethod
    def all_tags() -> list[str]:
        return [tag.text for tag in Tag.tags_list]


class Pattern:
    stemmed_words: list[str] = [] # all words
    def __init__(self, pattern_text: str, record_stem = True) -> None:
        
        # Properties
        self.pattern_text = pattern_text
        self.tokenize = NLP_Util.tokenize(pattern_text)
        self.stemmed_tokenize = [NLP_Util.stem(word) for word in self.tokenize if word != ""]

        # update stemmed words / all words
        for stem in self.stemmed_tokenize:
            if stem not in self.stemmed_words and record_stem:
                self.stemmed_words.append(stem)
        
    def __str__(self) -> str:
        return self.pattern_text
    
    @property
    def stemmed(self) -> list[str]:
        return self.stemmed_tokenize

    @property
    def in_bag_words(self) -> list[int]:
        bag = NLP_Util.bag_words(self.stemmed_tokenize, self.stemmed_words)
        return np.array(bag, dtype=np.float32)
        
from intent import *

# start initialing value
for intent in Intents:
    Tag(
        intent["tag"],
        intent["response"],
        *intent["patterns"]
    )


def Training_Data_XY() -> tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []
    for tag in Tag.tags_list:
        for pattern in tag.patterns:
            X.append(pattern.in_bag_words)
            Y.append(int(tag))
        
    return np.array(X), np.array(Y)
    
if __name__ == '__main__':
    X, Y = Training_Data_XY()