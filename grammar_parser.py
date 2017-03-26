import argparse
from collections import Counter


class Rule(object):
    """Represent a single Grammar rule."""

    def __init__(self, num, short, expanded, subsequence_starts, subsequence_lengths,
                 occurrence, use, min_length, max_length, mean_length):
        self.num = num
        self.short = short
        self.expanded = expanded
        self.subsequence_starts = subsequence_starts
        self.subsequence_lengths = subsequence_lengths
        self.occurrence = occurrence
        self.use = use
        self.min_length = min_length
        self.max_length = max_length
        self.mean_length = mean_length

    @property
    def words(self):
        return self.expanded.split() if self.expanded else []

    @property
    def flat(self):
        return self.expanded.replace(" ", "") if self.expanded else ""

    def __str__(self):
        return "%d -> %s" % (self.num, self.short)

    def __repr__(self):
        return self.__str__()


class GrammarParser(object):
    """Parse GrammarViz grammar files."""

    delim = " -> "

    def __init__(self, fpath):
        self.fpath = fpath
        self.enumerator = None
        self.line_num = 0
        self.line = None

    def parse(self):
        self._next_rule_num = 0
        with open(self.fpath) as f:
            self.enumerator = enumerate(f)
            grammar = self.parse_header()
            grammar.rule0 = self.parse_rule0()
            grammar.rules = self.parse_remaining_rules()

        return grammar

    def get_next_rule_num(self):
        next_num = self._next_rule_num
        self._next_rule_num += 1
        return next_num

    def parse_header(self):
        self.enumerator.next()  # discard filename line
        window_size = self.parse_next_option()
        paa_size = self.parse_next_option()
        alphabet_size = self.parse_next_option()
        return Grammar(window_size, paa_size, alphabet_size)

    def parse_remaining_rules(self):
        rules = []
        while True:
            try:
                rule = self.parse_next_rule()
                rules.append(rule)
            except StopIteration:
                break

        return rules

    def next_line(self):
        self.line_num, self.line = self.enumerator.next()
        return self.line

    def parse_next_option(self):
        line = self.next_line()
        return int(line.split()[-1])

    def parse_n_options(self, n):
        return [self.parse_next_option() for _ in range(n)]

    def parse_rule0(self):
        self.next_line()  # discard comment line
        line = self.next_line()
        short = line.split(self.delim)[1][1:-1]  # strip off quotes
        return Rule(self.get_next_rule_num(), short, None, [], [], *self.parse_n_options(5))

    def parse_next_rule(self):
        self.next_line()  # discard comment line
        line = self.next_line()
        part1, part2 = line.split(',')
        short = part1.split(self.delim)[1].replace("'", "")
        expanded = part2.split(':')[-1].replace("'", "").strip()

        line = self.next_line()
        starts = eval(line.split(': ')[1])
        line = self.next_line()
        lengths = eval(line.split(': ')[1])
        return Rule(self.get_next_rule_num(), short, expanded, starts, lengths,
                    *self.parse_n_options(5))


class Grammar(object):
    """Represent a grammar, including the options it was produced with."""

    def __init__(self, window_size, paa_size, alphabet_size):
        self.window_size = window_size
        self.paa_size = paa_size
        self.alphabet_size = alphabet_size

        self.rule0 = None
        self._rules = None
        self.rule_counts = Counter()
        self._rule_map = {}

    @staticmethod
    def from_file(fpath):
        parser = GrammarParser(fpath)
        return parser.parse()

    @property
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, rules):
        self._rules = rules
        self.rule_counts.clear()
        self._rule_map.clear()

        if rules is None:
            return

        for rule in rules:
            self.rule_counts[rule.num] = rule.occurrence
            self._rule_map[rule.num] = rule

    def topn(self, n):
        if self.rule_counts and self._rule_map:
            top = self.rule_counts.most_common(n)
            return [(self._rule_map[num], count) for num, count in top]
        else:
            return []

    def longest_rule(self):
        if not self.rules:
            return None
        return max(rule.expanded.count(" ") + 1 for rule in self.rules)

    def shortest_rule(self):
        if not self.rules:
            return None
        return min(rule.expanded.count(" ") + 1 for rule in self.rules)

    @property
    def tag(self):
        return "w%dp%da%d" % (self.window_size, self.paa_size, self.alphabet_size)

    def __str__(self):
        num_rules = len(self.rules) if self.rules else 0
        return "Grammar(window_size=%d, paa_size=%d, alphabet_size=%d) with %d rules" % (
            self.window_size, self.paa_size, self.alphabet_size, num_rules)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.rules) if self.rules else iter([])

    def __len__(self):
        return len(self.rules) if self.rules else 0

    def __getitem__(self, rule_num):
        return self._rule_map[rule_num]


def make_parser():
    parser = argparse.ArgumentParser(
        description="Parse grammar files into Grammar objects")
    parser.add_argument(
        "grammar_file",
        help="path to grammar file to parse")
    return parser


if __name__ == "__main__":
    cli_parser = make_parser()
    args = cli_parser.parse_args()

    try:
        grammar = Grammar.from_file(args.grammar_file)
    except Exception as e:
        print("Encountered error on Line %d: %s\nline: %s" % (
            parser.line_num, e, parser.line))
