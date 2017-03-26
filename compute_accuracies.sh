#!/bin/bash

echo "dataset1"
python motif_finder.py -tr data/dataset1/train.txt -te data/dataset1/test.txt -a 10 -p 3 -w 32
echo "dataset2"
python motif_finder.py -tr data/dataset2/train.txt -te data/dataset2/test.txt -a 8 -p 6 -w 32
echo "dataset3"
python motif_finder.py -tr data/dataset3/train.txt -te data/dataset3/test.txt -a 8 -p 5 -w 8
echo "dataset4"
python motif_finder.py -tr data/dataset4/train.txt -te data/dataset4/test.txt -a 8 -p 6 -w 16
echo "dataset5"
python motif_finder.py -tr data/dataset5/train.txt -te data/dataset5/test.txt -a 6 -p 4 -w 8
