from __future__ import annotations
import regex as re
from collections import defaultdict
from typing import Iterable, Iterator

def pre_tokenization(text: str, PAT: str, special_tokens: list[str], pre_tokens_str=None) -> dict[str]:
    i = 0
    if pre_tokens_str is None:
        pre_tokens_str = defaultdict(int)
    if special_tokens:
        st_PAT = "|".join([re.escape(x) for x in special_tokens])
        for st_match in re.finditer(st_PAT, text):
            j = st_match.start()
            ss = text[i:j]
            for match in re.finditer(PAT, ss):
                word = match.group()
                pre_tokens_str[word] += 1
            i = st_match.end()
    ss = text[i:]
    for match in re.finditer(PAT, ss):
        word = match.group()
        pre_tokens_str[word] += 1
    return pre_tokens_str


# update bp count and pre_tokens as a result of bp merge
def update_bp_count(bp, bp_count, pre_tokens):
    for ini_word in list(pre_tokens.keys()):
        i = 0
        count = pre_tokens[ini_word]
        word = ini_word
        # scan through the word to look for the bp. if found, update the word and bp_count influenced by the merge
        while i < len(word)-1:
            if word[i]==bp[0] and word[i+1]==bp[1]:
                if i > 0:
                    bp_count[(word[i-1], bp[0]+bp[1])] += count
                    bp_count[(word[i-1], bp[0])] -= count
                if i+2 < len(word):
                    bp_count[(bp[0]+bp[1], word[i+2])] += count
                    bp_count[(bp[1], word[i+2])] -= count
                bp_count[(bp[0], bp[1])] -= count
                word = word[:i] + (bp[0]+bp[1],) + word[i+2:]
            i += 1

        # update tokenization of the pretoken / word
        if word != ini_word:
            pre_tokens.pop(ini_word)
            pre_tokens[word] = count


# update bp count and pre_tokens as a result of bp merge
def _update_bp_count(bp, bp_count, pre_tokens):
    for ini_word in list(pre_tokens.keys()):
        i = 0
        count = pre_tokens[ini_word]
        word = ini_word
        # scan through the word to look for the bp. if found, update the word and bp_count influenced by the merge
        while i < len(word)-1:
            if word[i]==bp[0] and word[i+1]==bp[1]:
                if i > 0:
                    bp_count[(word[i-1], bp[0]+bp[1])] += count
                    bp_count[(word[i-1], bp[0])] -= count
                if i+2 < len(word):
                    bp_count[(bp[0]+bp[1], word[i+2])] += count
                    bp_count[(bp[1], word[i+2])] -= count
                bp_count[(bp[0], bp[1])] -= count
                word = word[:i] + (bp[0]+bp[1],) + word[i+2:]
            i += 1

        # update tokenization of the pretoken / word
        if word != ini_word:
            pre_tokens.pop(ini_word)
            pre_tokens[word] = count
            


def construct_bpe(pre_tokens_str: dict[str],  vocab_size: int, special_tokens: list[str]):
    pre_tokens = dict()
    for word, count in pre_tokens_str.items():
        word_tuple_of_bytes = tuple(bytes([a]) for a in word.encode("utf-8"))
        pre_tokens[word_tuple_of_bytes] = count
    
    # construct initial vocab
    token_id = 0
    vocab = dict()
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    
    if special_tokens:
        for t in special_tokens:
            vocab[token_id] = t.encode("utf-8")
            token_id += 1
    

    # construct initial bytes-pair count
    bp_count = defaultdict(int)
    for token, count in pre_tokens.items():
        for i in range(len(token)-1):
            bp = (token[i], token[i+1])
            bp_count[bp] += count

    # merge
    merges = []
    while len(vocab) < vocab_size:
        # find the most frequent bp, if there is a tie then pick the first based on lexi order
        max_count = max(bp_count.values())
        tmp = [bp for bp, count in bp_count.items() if count==max_count]
        bp = max(tmp)
        merges.append(bp)

        # add it to vocab
        vocab[token_id] = bp[0] + bp[1]
        token_id += 1

        # update bp count and tokenization of pre-tokens
        _update_bp_count(bp, bp_count, pre_tokens)
        assert(bp_count[bp]==0)
        
    return vocab, merges




class Tokenizer():
    def __init__(self, vocab, merges, special_tokens, PAT=None):
        # bidirectional mapping between ids and tokens
        self.id2token = vocab
        self.token2id = defaultdict(None)
        for i, token in self.id2token.items():
            self.token2id[token] = i
        
        # ordered set of merges for fast query
        self.merges = dict()
        for i in range(len(merges)):
            self.merges[merges[i]] = i
        
        # check whether special token list is empty
        if special_tokens:
            special_tokens.sort(key=len, reverse=True) # sort special token list in descending-in-length order
            self.st_PAT = "|".join([re.escape(x) for x in special_tokens])
        else:
            self.st_PAT = None
        self.special_tokens = special_tokens

        if PAT:
            self.PAT = PAT
        else:
            self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def _encode_pretoken(self, bs: bytes):
        if bs in self.token2id:
            return [self.token2id[bs]]
        
        word = tuple(bytes([b]) for b in bs)

        while True:
            matches = set()
            for i in range(len(word)-1):
                bp = (word[i], word[i+1])
                if bp in self.merges:
                    matches.add((self.merges[bp], bp, i))
            if not matches:
                break
            tmp = min(matches)
            _, bp, i = tmp
            word = word[:i] + (bp[0]+bp[1],) + word[i+2:]
        
        return [self.token2id[t] for t in word]
    
    
    # turn a string without special token into pre-token list
    def encode(self, s : str) -> list[int]:
        result = []

        i = 0
        if self.special_tokens:
            for st_match in re.finditer(self.st_PAT, s):
                j = st_match.start()
                ss = s[i:j]
                for match in re.finditer(self.PAT, ss):
                    bs = match.group().encode("utf-8")
                    result.extend(self._encode_pretoken(bs))
                st_bs = st_match.group().encode("utf-8")
                result.append(self.token2id[st_bs])
                i = st_match.end()
        ss = s[i:]
        for match in re.finditer(self.PAT, ss):
            bs = match.group().encode("utf-8")
            result.extend(self._encode_pretoken(bs))
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            id_list = self.encode(s)
            for i in id_list:
                yield i

    
    def decode(self, token_ids: Iterable[int]) -> str:
        text = b"".join(self.id2token[i] for i in token_ids).decode("utf-8", errors='replace')
        return text