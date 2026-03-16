import json
import re
class BPETokenizer:
    def __init__(self, vocab_size=None,min_freq=2, special_tokens=None):
        self.special_tokens = ['<PAD>','<UNK>','<SOS>','<EOS>'] if special_tokens is None else special_tokens
        self.vocab_size = vocab_size if vocab_size is not None else 1000
        self.min_freq = min_freq
        self.token_to_id = {}
        self.id_to_token = {}
        self.merged_tokens = {}
        self.merged_tokens_rank = {}
        
    def _pretokenize(self, text):
        raw_tokens = re.findall(r'\s|[^\s]+', text)
        tokens = []
        pending_spaces = 0
        
        for token in raw_tokens:
            if token == " ":
                pending_spaces += 1
                continue
            if pending_spaces:
                token = ('\u0120' * pending_spaces) + token
                pending_spaces = 0
            tokens.append(token)
            
        if pending_spaces:
            tokens.append('\u0120' * pending_spaces)
            
        return tokens
    
    def train(self, corpus):
        #1 create vocab from corpus
        vocab_word_freq = {}
        for text in corpus:
            tokens = self._pretokenize(text)
            for word_punc in tokens:
                if word_punc == ' ':
                    word_punc = '\u0120'
                if word_punc not in vocab_word_freq:
                    vocab_word_freq[word_punc] = 0
                vocab_word_freq[word_punc] += 1
        #2 initialize token_to_id and id_to_token with special tokens
        index =0
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            index += 1
        
        #3 initialize token_to_id and id_to_token with characters in vocab
        for word in vocab_word_freq.keys():
            for char in word:
                if char not in self.token_to_id:
                    self.token_to_id[char] = index
                    self.id_to_token[index] = char
                    index += 1
        word_split = {word: list(word) for word in vocab_word_freq.keys()}
        #4 freq for bpe
        vocab_token_freq = {}
        pair_to_words = {}
        for word, freq in vocab_word_freq.items():
            tokens = list(word)
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                vocab_token_freq[pair] = vocab_token_freq.get(pair, 0) + freq
                
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word)
        
        #5 bpe merge
        rank = 0
        while (len(self.token_to_id) < self.vocab_size):
            if not vocab_token_freq:
                print(f"Early termination: No more pairs to merge. Final vocab size: {len(self.token_to_id)}")
                break
            
            # find the most frequent pair breaking ties by lexicographical order
            most_freq_pair = min(vocab_token_freq, key=lambda x: (-vocab_token_freq[x], x))
            max_freq = vocab_token_freq[most_freq_pair]
            
            if max_freq < self.min_freq:
                print(f"Early termination: Highest pair frequency ({max_freq}) dropped below threshold ({self.min_freq}). Final vocab size: {len(self.token_to_id)}")
                break
            new_token = ''.join(most_freq_pair)
            self.token_to_id[new_token] = index
            self.id_to_token[index] = new_token
            index += 1
            self.merged_tokens[new_token] = most_freq_pair
            self.merged_tokens_rank[new_token] = rank
            rank += 1
            
            words_to_update = pair_to_words.get(most_freq_pair, set()).copy()
            if most_freq_pair in vocab_token_freq:
                del vocab_token_freq[most_freq_pair]
            if most_freq_pair in pair_to_words:
                del pair_to_words[most_freq_pair]
            #update word_split and vocab_token_freq
            for word in words_to_update:
                freq = vocab_word_freq[word]
                old_tokens = word_split[word]
                for i in range(len(old_tokens)-1):
                    pair = (old_tokens[i], old_tokens[i+1])
                    if pair != most_freq_pair and pair in vocab_token_freq:
                        vocab_token_freq[pair] -= freq
                        if vocab_token_freq[pair] <= 0:
                            del vocab_token_freq[pair]
                            if pair in pair_to_words and word in pair_to_words[pair]:
                                pair_to_words[pair].remove(word)
                
                i = 0
                new_tokens = []
                while i < len(old_tokens):
                    if i < len(old_tokens) - 1 and (old_tokens[i], old_tokens[i+1]) == most_freq_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(old_tokens[i])
                        i += 1
                word_split[word] = new_tokens
                
      
                for i in range(len(new_tokens)-1):
                    pair = (new_tokens[i], new_tokens[i+1])
                    vocab_token_freq[pair] = vocab_token_freq.get(pair, 0) + freq
                    if pair not in pair_to_words:
                        pair_to_words[pair] = set()
                    pair_to_words[pair].add(word)
            
        
    
    def encode(self, text):
        words = self._pretokenize(text)
        token_ids = []
        word_to_word_split = {}
        for word in words:
            char_list = []
            for char in word:
                if char == ' ':
                    char = '\u0120'
                if char not in self.token_to_id:
                    char = '<UNK>'
                char_list.append(char)
            
            while(True):
                available_pairs = {}
                for i in range(len(char_list)-1):
                    pair = (char_list[i], char_list[i+1])
                    pair_str = ''.join(pair)
                    if pair_str in self.merged_tokens:
                        available_pairs[self.merged_tokens_rank[pair_str]] = pair
                if available_pairs!={}:
                    best_pair = available_pairs[min(available_pairs.keys())]
                    new_char_list = []
                    i = 0
                    while i < len(char_list):
                        if i < len(char_list) - 1:
                            pair = (char_list[i], char_list[i+1])
                            if pair == best_pair:
                                new_char_list.append(''.join(pair))
                                i += 2
                            else:
                                new_char_list.append(char_list[i])
                                i += 1
                        else:
                            new_char_list.append(char_list[i])
                            i += 1
                    char_list = new_char_list
                else:
                    break
            word_to_word_split[word] = char_list
            for token in char_list:
                if token not in self.token_to_id:
                    token_id = self.token_to_id['<UNK>']
                else:
                    token_id = self.token_to_id[token]
                token_ids.append(token_id)
        return token_ids
                    
    def decode(self, token_ids):
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        text = "".join(tokens).replace('\u0120', ' ')
        return text
        

    def save(self, filepath):
        state = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "merged_tokens": self.merged_tokens,     
            "merged_tokens_rank": self.merged_tokens_rank
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.vocab_size = state["vocab_size"]
        self.special_tokens = state["special_tokens"]
        self.token_to_id = state["token_to_id"]
        self.merged_tokens_rank = state["merged_tokens_rank"]
        
        #handling json messup as keys are always strings adn no tuples
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.merged_tokens = {k: tuple(v) for k, v in state["merged_tokens"].items()}
    def get_vocab_size(self):
        return len(self.token_to_id)
    
    def get_unk_id(self):
        return self.token_to_id['<UNK>']
