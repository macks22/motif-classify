import logging
import argparse

import numpy as np


def read_data(train_path, test_path):
    logging.info("Reading training data from: " + train_path)
    train = np.loadtxt(train_path)
    logging.info("Reading test data from: " + test_path)
    test = np.loadtxt(test_path)

    X_train, y_train = train[:, 1:], train[:, 0].astype(np.int)
    logging.info("Train data shape: %d X %d" % X_train.shape)
    X_test, y_test = test[:, 1:], test[:, 0].astype(np.int)
    logging.info("Test data shape: %d X %d" % X_test.shape)
    
    return X_train, y_train, X_test, y_test


def make_parser():
    """Create CLI parser."""
    parser = argparse.ArgumentParser("Get stats from time series dataset files")
    parser.add_argument(
        "-tr", "--train",
        default="train.txt",
        help="path to training data file")
    parser.add_argument(
        "-te", "--test",
        default="test.txt",
        help="path to test data file")
    parser.add_argument(
        "-v", "--verbosity",
        type=int, default=1,
        help="verbosity level for logging; default=1 (INFO)")
    return parser


def parse_args(parser):
    """Parse arguments using the given parser and configure logging.
    :returns: args parsed.

    """
    args = parser.parse_args()
    reload(logging)
    logging.basicConfig(
        level=(logging.INFO if args.verbosity == 1
               else logging.DEBUG if args.verbosity > 1
               else logging.ERROR),
        format="[%(levelname)s][%(asctime)s]: %(message)s")
    return args
