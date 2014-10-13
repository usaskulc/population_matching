Install
=======
Method for matching a subsample to a representative sample from a larger population.
Install dependencies from pip -r requirements.txt

Example Usage
=============
Imagine a sample datafile that had a special study group as a treatement.  One might
match this pairwise against the general population using:

python resample.py -i example_datafile.csv -c example_definitions.csv -m "Study Group" -o out.csv
python resample.py -i ../data_lak_2015/input-new.csv -c ../data_lak_2015/definitions-new.csv -m "treat" -o out.csv

