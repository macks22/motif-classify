# Installation

## GrammarViz

My code wraps up code from GrammarViz. To use it, you'll first need to run:

    git clone https://github.com/GrammarViz2/grammarviz2_src.git
    cd grammarviz2_src
    mvn package -Psingle

...to build the jar. I wrote a wrapper that implements the scikit-learn
`BaseEstimator` interface so I could use scikit's `Pipeline` and `GridSearchCV`
for cross-validation parameter grid search. To allow the wrapper to find the
GrammarViz code, you have two options. You can either add an environment
variable, like so:

    export GRAMMARVIZ_HOME="/path/to/grammarviz2_src/target/"

...or you can pass that path using the `--gviz-home` flag.

## My Code

The other dependencies are listed in the `requirements.txt` file. Note that
I've bundled one of the requirements (`pysax`) since the author recommended
this. All custom code is in pure Python, so there is no need to build anything.

# Code Outline and Execution

The custom Python code is contained in 4 modules:

    grammar_parser.py -- Parse text output from GrammarViz CLI into `Grammar` object
    motif_tagger.py -- SAX-discretize time series and use `Grammar` object to tag with motifs
    motif_finder.py -- Provide a convenient method to run the GrammarViz code and parse it
                       directly into Python using temporary files for IPC. The code spawns
                       a subprocess shell to run the JVM in, then parses the results back
                       into the Python `Grammar` object. This code contains the interface
                       for the `sklearn.BaseEstimator`, called `MotifFinder`.
    grid_search.py -- Using the `MotifFinder` for feature selection and a `sklearn.RandomForest`
                      for classification, conduct a grid search over the SAX parameters. The data is output to a csv file and the accuracy of the best estimator is output to stdout.

Both `motif_finder.py` and `grid_search.py` share a common base CLI.
You can see this by running:

    python motif_finder.py -h

I've included the output here for reference:

	usage: Get stats from time series dataset files [-h] [-tr TRAIN] [-te TEST]
													[-v VERBOSITY]
													[-w WINDOW_SIZE] [-p PAA_SIZE]
													[-a ALPHABET_SIZE]
													[-g GVIZ_HOME]

	Motif finding using sequitur

	optional arguments:
	  -h, --help            show this help message and exit
	  -tr TRAIN, --train TRAIN
							path to training data file
	  -te TEST, --test TEST
							path to test data file
	  -v VERBOSITY, --verbosity VERBOSITY
							verbosity level for logging; default=1 (INFO)
	  -w WINDOW_SIZE, --window-size WINDOW_SIZE
							window size
	  -p PAA_SIZE, --paa-size PAA_SIZE
							number of letters in a word; SAX word size
	  -a ALPHABET_SIZE, --alphabet-size ALPHABET_SIZE
							size of alphabet; number of bins for SAX
	  -g GVIZ_HOME, --gviz-home GVIZ_HOME
							path to directory with gviz jar

The `grid_search.py` module also contains an option for where to output results to:

    -o OUTPUT_PATH, --output-path OUTPUT_PATH
                          path to write CV scores to

## Example Data and Scripts

Finally, I've included two bash scripts and sample data that I used to run some
experiments using this code. These scripts assume you have the data in the
following structure (same as in zip archive):

    ├── dataset1
    │   ├── test.txt
    │   └── train.txt
    ├── dataset2
    │   ├── test.txt
    │   └── train.txt
    ├── dataset3
    │   ├── test.txt
    │   └── train.txt
    ├── dataset4
    │   ├── test.txt
    │   └── train.txt
    └── dataset5
        ├── test.txt
        └── train.txt

To run the grid search on all datasets, use `grid_search.sh`. To compute the
accuracies using the best parameters I found in my experiments, run
`compute_accuracies.sh`.
