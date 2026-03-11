class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.special_tokens = ['<PAD>','<UNK>','<SOS>','<EOS>'] if special_tokens is None else special_tokens
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merged_tokens = {}
        self.merged_tokens_rank = {}
        

    def train(self, corpus):
        #1 create vocab from corpus
        vocab_word_freq = {}
        for text in corpus:
            new_text = text.replace(' ', ' \u0120')
            for word in new_text.split():
                if word not in vocab_word_freq:
                    vocab_word_freq[word] = 0
                vocab_word_freq[word] += 1
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
        for word, freq in vocab_word_freq.items():
            tokens = list(word)
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                if pair not in vocab_token_freq:
                    vocab_token_freq[pair] = 0
                vocab_token_freq[pair] += freq
        
        #5 bpe merge
        rank = 0
        while (len(self.token_to_id) < self.vocab_size):
            if not vocab_token_freq:
                break
            # find the most frequent pair breaking ties by lexicographical order
            most_freq_pair = min(vocab_token_freq, key=lambda x: (-vocab_token_freq[x], x))
            new_token = ''.join(most_freq_pair)
            self.token_to_id[new_token] = index
            self.id_to_token[index] = new_token
            index += 1
            self.merged_tokens[new_token] = most_freq_pair
            self.merged_tokens_rank[new_token] = rank
            rank += 1
            #update word_split and vocab_token_freq
            for word, tokens in word_split.items():
                i = 0
                new_tokens=[]
                while i < len(tokens):
                    if i < len(tokens) - 1:
                        pair = (tokens[i], tokens[i+1])
                        if pair == most_freq_pair:
                            new_tokens.append(new_token)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_split[word] = new_tokens
            
            vocab_token_freq = {}
            for word, freq in vocab_word_freq.items():
                tokens = word_split[word]
                for i in range(len(tokens)-1):
                    pair = (tokens[i], tokens[i+1])
                    if pair not in vocab_token_freq:
                        vocab_token_freq[pair] = 0
                    vocab_token_freq[pair] += freq
            
        
    
    def encode(self, text):
        raise NotImplementedError("Encoding method not implemented yet.")

    def decode(self, token_ids):
        raise NotImplementedError("Decoding method not implemented yet.")

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
    
    def get_vocab_size(self):
        raise NotImplementedError("Get vocab size method not implemented yet.")
    
    def get_unk_id(self):
        raise NotImplementedError("Get unk id method not implemented yet.")
