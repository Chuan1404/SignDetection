from collections import Counter
import json

class Vocabulary:
    def __init__(self, min_freq=1, specials=None):
        if specials is None:
            specials = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

        self.min_freq = min_freq
        self.specials = specials

        self.word2idx = {}
        self.idx2word = {}
        self.counter = Counter()

        # add special tokens first
        for token in specials:
            self._add_word(token)

    def _add_word(self, word):
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx] = word

    def build_vocab(self, sentences):
        """
        sentences: list of strings
        Example:
        [
            "I love deep learning",
            "Sign language translation is interesting"
        ]
        """

        # count words
        for sentence in sentences:
            tokens = sentence.lower().split()
            self.counter.update(tokens)

        # add words above min frequency
        for word, freq in self.counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self._add_word(word)

    def encode(self, sentence):
        tokens = sentence.lower().split()

        encoded = [self.word2idx["<SOS>"]]

        for token in tokens:
            encoded.append(
                self.word2idx.get(token, self.word2idx["<UNK>"])
            )

        encoded.append(self.word2idx["<EOS>"])

        return encoded

    def decode(self, indices):
        words = []

        for idx in indices:
            word = self.idx2word.get(idx, "<UNK>")

            if word in ["<SOS>", "<EOS>", "<PAD>"]:
                continue

            words.append(word)

        return " ".join(words)

    def save(self, path):
        data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "min_freq": self.min_freq
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.min_freq = data["min_freq"]


if __name__ == "__main__":

    # example sentences
    sentences = [
        "I love sign language",
        "Sign language translation is important",
        "Deep learning helps translation"
    ]

    # build vocab
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(sentences)

    print("Vocabulary Size:", len(vocab.word2idx))
    print(vocab.word2idx)

    # encode
    sentence = "sign language translation"
    encoded = vocab.encode(sentence)

    print("Encoded:", encoded)

    # decode
    decoded = vocab.decode(encoded)
    print("Decoded:", decoded)

    # save vocab
    vocab.save("vocab.json")

    # load vocab
    new_vocab = Vocabulary()
    new_vocab.load("vocab.json")

    print("Loaded Vocabulary:")
    print(new_vocab.word2idx)
