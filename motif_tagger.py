import string
from collections import defaultdict

import pysax

from grammar_parser import Grammar


class MotifTagger(object):
    """Lookup motifs in a prefix tree using SAX substrings."""

    def __init__(self, grammar):
        self.grammar = grammar
        self.sax = pysax.SAXModel(
            window=grammar.window_size,
            stride=1,
            nbins=grammar.paa_size,
            alphabet=string.lowercase[:grammar.alphabet_size])

    @staticmethod
    def from_grammar_file(fpath):
        grammar = Grammar.from_file(fpath)
        return MotifTagger(grammar)

    @property
    def grammar(self):
        return self._grammar

    @grammar.setter
    def grammar(self, grammar):
        self._build_map(grammar)
        self._longest = grammar.longest_rule()
        self._shortest = grammar.shortest_rule()
        self._grammar = grammar

    def _build_map(self, grammar):
        self._prefix_tree = PrefixWordMap()
        for rule in grammar:
            self._prefix_tree[rule.words] = rule.num

    def tag_ts(self, ts):
        """Tag a time series with motifs."""
        words = self.sax.symbolize_signal(ts)
        return self.tag(words)

    def tag(self, words):
        """Tag a SAX-discretized time series with motifs from a grammar.
        @param words: list of words representing a SAX-discretized time series.

        """
        num_words = self._longest
        shortest = self._shortest
        rule_nums = []
        for i in range(len(words)):
            subseq = words[i: i + num_words]
            rule_nums += self._prefix_tree.get_all(subseq, shortest)

        return rule_nums

    def __getitem__(self, words):
        return self._prefix_tree[words]


class PrefixWordMap(object):
    """Store values for SAX word sequences, with efficient lookup for prefixes."""

    def __init__(self):
        self.prefix_tree = defaultdict(PrefixWordMap)
        self.values = []

    def get_all(self, words, shortest=1):
        """Get all values associated with all prefix subsets of the words list."""
        values = []
        prefix = words[:shortest]
        node = self
        for word in prefix:
            node = node.prefix_tree[word]
            values += node.values

        if not node.prefix_tree:
            return values

        for word in words[shortest:]:
            node = node.prefix_tree[word]
            values += node.values
            if not node.prefix_tree:
                return values

        return values

    def __getitem__(self, words):
        node = self
        if hasattr(words, '__iter__'):
            for word in words:
                node = node.prefix_tree[word]
        else:
            node = self.prefix_tree[words]

        return node.values

    def __setitem__(self, words, value):
        """Add a new word sequence to the map."""
        if not words:
            raise ValueError("Key must be valid word sequence")

        node = self
        for word in words:
            node = node.prefix_tree[word]

        node.values.append(value)

    def __contains__(self, words):
        node = self
        for word in words:
            if not node.prefix_tree.has_key(word):
                return False
            node = node.prefix_tree[word]

        return True

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return self.__str__()


class MultiGrammarMotifTagger(object):
    """Tag time series data with motifs from several grammars."""

    def __init__(self, grammars):
        """
        @param grammars: a list of Grammar objects to use for tagging.

        """
        self.grammars = {}
        for grammar in grammars:
            self.grammars[grammar.tag] = grammar

    def tag_ts(self, ts):
        """Tag a time series with motifs from multiple grammars.
        @param ts: The time series, as an iterable of floats.

        """
        tags = []
        for grammar_tag, grammar in self.grammars.items():
            tags += [grammar_tag + tag for tag in grammar.tag_ts(ts)]
        return tags


if __name__ == "__main__":
    from grammar_parser import make_parser
    parser = make_parser()
    args = parser.parse_args()

    tagger = MotifTagger.from_grammar_file(args.grammar_file)

    rules = tagger.grammar.rules[:5]
    words = []
    expected = []
    for rule in rules:
        expected.append(rule.num)
        words += rule.words
        words.append('aaaa')

    tags = tagger.tag(words)
