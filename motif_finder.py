"""
Use GrammarViz to find motifs via Sequitur.

"""
import os
import uuid
import logging
import itertools
import subprocess

import pathos.multiprocessing as mp
import numpy as np
import scipy.sparse as sps

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn import metrics

from motif_tagger import MotifTagger
import cli


GVIZ_HOME = 'GRAMMARVIZ_HOME'
JAR_NAME = "grammarviz2-0.0.1-SNAPSHOT-jar-with-dependencies.jar"


def delete_silently(fpath):
    """Delete a file if possible, else do nothing."""
    try:
        os.remove(fpath)
    except OSError:
        pass


class MotifFinder(TransformerMixin, BaseEstimator):
    """Find motifs using GrammarViz executable."""

    def __init__(self, window_size=8, paa_size=4, alphabet_size=4, strategy='exact',
                 n_jobs=-1, gviz_home=None):
        if not gviz_home:
            try:
                gviz_home = os.environ[GVIZ_HOME]
            except KeyError:
                pass

        if gviz_home is None:
            raise ValueError("Must configure env variable '%s' or override" % GVIZ_HOME)

        self.jar_path = os.path.join(gviz_home, JAR_NAME)

        self.window_size = window_size
        self.paa_size = paa_size
        self.alphabet_size = alphabet_size
        self.strategy = strategy

        self.command = (
            "java -cp {jar_path} net.seninp.grammarviz.cli.TS2SequiturGrammar"
            " -d {input_path}"
            " -o {output_path}"
            " -w {window_size}"
            " -p {paa_size}"
            " -a {alphabet_size}"
            " --strategy {strategy}"
            " --num-workers {n_jobs}"
            " --prune"
        )

        self.tagger = None
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count() - 1
        else:
            self.n_jobs = n_jobs if n_jobs > 0 else 1

    def get_command_args(self):
        n_jobs = self.n_jobs + 1 if self.n_jobs != 1 else 1
        params = self._get_params()
        params.update({
            'jar_path': self.jar_path,
            'n_jobs': n_jobs
        })
        return params

    def _get_params(self):
        return {
            'window_size': self.window_size,
            'paa_size': self.paa_size,
            'alphabet_size': self.alphabet_size,
            'strategy': self.strategy
        }

    def fit(self, X, y=None):
        tmp_output_fname = os.path.abspath(str(uuid.uuid4()))
        tmp_input_fname = os.path.abspath(str(uuid.uuid4()))
        logging.info("Writing flattened data to tmp file: %s" % tmp_input_fname)
        np.savetxt(tmp_input_fname, X.flat)

        args = self.get_command_args()
        args.update({
            'input_path': tmp_input_fname,
            'output_path': tmp_output_fname
        })
        command = self.command.format(**args)
        logging.info("Executing command: %s" % command)

        try:
            proc = subprocess.Popen(command, shell=True)
            status = proc.wait()
            if status != 0:
                raise OSError("Sequitur routine exited with non-0 status code: %d" % status)
            self.tagger = MotifTagger.from_grammar_file(tmp_output_fname)
        finally:
            delete_silently(tmp_input_fname)
            delete_silently(tmp_output_fname)

        logging.info("Successfully discovered %d motifs" % len(self.tagger.grammar))
        return self

    def transform(self, X):
        if self.tagger is None:
            raise ValueError("Must find_motifs before you can tag anything")

        logging.info("Tagging %s data with motifs using %d workers..." % (
            str(X.shape), self.n_jobs))

        if self.n_jobs > 1:
            pool = mp.ProcessingPool(self.n_jobs)
            splits = np.array_split(X, self.n_jobs)
            tag_lists = pool.map(self._tag_motifs, splits)
            tags = list(itertools.chain.from_iterable(tag_lists))
        else:
            tags = self._tag_motifs(X)

        logging.info("All motifs have been tagged")
        return self._sparsify_tags(tags)

    def _sparsify_tags(self, rows):
        counts = np.array([len(row) for row in rows])
        col_indices = np.concatenate(rows)
        row_indices = np.repeat(np.arange(counts.shape[0]), counts)
        data = np.ones(col_indices.shape[0])
        n_features = len(self.tagger.grammar)
        return sps.csr_matrix((data, (row_indices, col_indices - 1)),
                              shape=(len(rows), n_features))

    def _tag_motifs(self, X):
        return [self.tagger.tag_ts(row) for row in X]


def make_parser():
    parser = cli.make_parser()
    parser.description = "Motif finding using sequitur"
    parser.add_argument(
        '-w', '--window-size',
        type=int, default=8,
        help='window size')
    parser.add_argument(
        '-p', '--paa-size',
        type=int, default=4,
        help='number of letters in a word; SAX word size')
    parser.add_argument(
        '-a', '--alphabet-size',
        type=int, default=4,
        help='size of alphabet; number of bins for SAX')
    parser.add_argument(
        '-g', '--gviz-home',
        default="/Users/msweeney/workshop/cs674/grammarviz2_src/target/",
        help='path to directory with gviz jar')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = cli.parse_args(parser)
    X_train, y_train, X_test, y_test = cli.read_data(args.train, args.test)

    finder = MotifFinder(
        gviz_home=args.gviz_home, window_size=args.window_size,
        paa_size=args.paa_size, alphabet_size=args.alphabet_size)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=mp.cpu_count() - 2)
    pipeline = make_pipeline(finder, clf)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.4f" % accuracy)
