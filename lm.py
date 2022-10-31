import collections

class Uniform(object):
    """Uniform distribution."""
    def __init__(self, data):
        vocab = {"<unk>", "</s>"}
        for words in data:
            vocab.update(words)
        self.vocab = vocab
    def prob(self, u, w):
        return 1/len(self.vocab)

class KneserNey(object):
    def __init__(self, data, n, bom=None):
        self.bom = bom
        self.order = n
        
        # Collect n-gram counts
        cuw = collections.defaultdict(collections.Counter)
        cu = collections.Counter()
        for line in data:
            u = ("<s>",)*(n-1)
            for w in line + ["</s>"]:
                cuw[u][w] += 1
                cu[u] += 1
                u = (u+(w,))[1:]

        # Compute discount
        cc = collections.Counter()
        for u in cuw:
            for w in cuw[u]:
                cc[cuw[u][w]] += 1
        d = cc[1] / (cc[1] + 2*cc[2])

        # Compute probabilities and backoff weights
        self._prob = collections.defaultdict(dict)
        self._bow = {}
        for u in cuw:
            for w in cuw[u]:
                self._prob[u][w] = (cuw[u][w]-d) / cu[u]
            self._bow[u] = len(cuw[u])*d / cu[u]

    def prob(self, u, w):
        if u in self._prob:
            return self._prob[u].get(w, 0) + self._bow[u] * self.bom.prob(u[1:], w)
        else:
            return self.bom.prob(u[1:], w)
