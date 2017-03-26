import multiprocessing as mp

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd

import cli
import motif_finder


def make_parser():
    parser = motif_finder.make_parser()
    parser.description = "Grid search for best SAX params for classification"
    parser.add_argument(
        '-o', '--output-path',
        default='',
        help='path to write CV scores to')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = cli.parse_args(parser)
    X_train, y_train, X_test, y_test = cli.read_data(args.train, args.test)

    finder = motif_finder.MotifFinder(gviz_home=args.gviz_home)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=mp.cpu_count() - 2)
    pipeline = make_pipeline(finder, clf)

    param_grid = {
        'motiffinder__window_size': [8, 16, 32, 64],
        'motiffinder__paa_size': [3, 4, 5, 6],
        'motiffinder__alphabet_size': [6, 8, 10]
    }
    score_func = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
    cv_search = GridSearchCV(pipeline, cv=3, param_grid=param_grid,
                             scoring=score_func, verbose=10, n_jobs=2)

    cv_search.fit(X_train, y_train)
    y_pred = cv_search.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.4f" % accuracy)

    if args.output_path:
        rows = [
            (tup.mean_validation_score, tup.cv_validation_scores.std(),
            tup.parameters['motiffinder__alphabet_size'],
            tup.parameters['motiffinder__paa_size'],
            tup.parameters['motiffinder__window_size'])
            for tup in cv_search.grid_scores_]
        output = pd.DataFrame(
            rows, columns=['mean', 'std', 'alphabet_size', 'paa_size', 'window_size'])
        output.to_csv(args.output_path)
