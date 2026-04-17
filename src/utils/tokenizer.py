
class Tokenizer:
    def __init__(self):
        self.word2idx = {
            "<pad>":0,
            "<bos>":1,
            "<eos>":2,
            "<unk>":3
        }
        self.idx2word = {v:k for k,v in self.word2idx.items()}

    def build_vocab(self, texts):
        idx = len(self.word2idx)

        for text in texts:
            for w in text.lower().split():
                if w not in self.word2idx:
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
                    idx += 1

    def encode(self, text):
        tokens = [1]

        for w in text.lower().split():
            tokens.append(self.word2idx.get(w, 3))

        tokens.append(2)

        return tokens

    def decode(self, ids):
        words = []

        for i in ids:
            if i == 2:
                break
            if i > 3:
                words.append(self.idx2word[i])

        return " ".join(words)

    @property
    def vocab_size(self):
        return len(self.word2idx)