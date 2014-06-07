# sudo pip install hungarian
# sudo easy_install statsmodels
#sudo apt-get install python-pandas

import pandas as pd
import numpy as np
import hungarian
import math
import argparse
from multiprocessing import Pool
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy.stats import chisquare

from sklearn import linear_model
import sys

df_full = None
df_columns = None

column_parameters = {}
column_funs = {}
column_weights = {}


def print_report():
    #generate an output report
    print "Column weights used:"
    for attribute in sorted(column_weights.keys()):
        print "\t" + attribute.ljust(30) + "\t" + str(column_weights[attribute])
    print""
    print "Two-sample Kolmogorov-Smirnov:"
    ks_vals = ks(df_sample_condition, df_matches)
    for attribute in sorted(ks_vals.keys()):
        if ks_vals[attribute][1] < 0.1:
            print "*",
        print "\t" + attribute.ljust(30) + "\tD={:.4f}\tp={:.4f}".format(*ks_vals[attribute])
    print ""
    print "Independent two-sample t-test:"
    for attribute in sorted(df_columns.columns):
        if df_columns.ix[0][attribute] == "ignore":
            try:
                t, p = ttest_ind(df_sample_condition[attribute], df_matches[attribute])
                print "\t" + attribute.ljust(30) + "\tt={:.4f}\tp={:.4f}".format(t, p)
            except:
                pass  #oops, must not have been an integer value!


def ks(df_sample_condition, df_match):
    """For every column that has not been set to ignore and is not the match_attribute
    this function will perform a two sample Kolmogorov-Smirnov test
    """
    ks_results = {}
    for column in df_columns:
        if df_columns.ix[0][column] != "ignore":
            ks_results[column] = ks_2samp(df_sample_condition[column], df_match[column])
    return ks_results


def write_output(output, df_sample_condition, df_matches, side_by_side=False):
    """Writes the sample condition and matches to a single csv file.  If side_by_side
    then the sample condition is written to the left of the matches, with pairs matched
    appropriately.  Otherwise, the sample condition is written above the matches,
    with the matched pairs in order.
    """
    if (side_by_side):
        columns_to_rename = {}
        for column in df_sample_condition.columns:
            columns_to_rename[column] = "matched_" + str(column)
        df_output = df_matches.rename(columns=columns_to_rename)
        df_output = df_output.rename(columns={"_index": "index"})  #rename the index we will join on back
        df_output = df_sample_condition.join(df_output)
    else:
        df_output = pd.concat([df_sample_condition, df_matches])
    df_output.to_csv(file_output, encoding='utf-8')


def run_hungarian(matrix, df_population, df_sample_condition):
    """Runs the hungarian linear assignment problem solver from the hungarian package.
    Takes in a matrix of datavalues and dataframes for the df_population and the
    df_sample_condition.  Returns the matches as a new dataframe with the same
    structure as the df_population.
    """
    row_assigns, col_assigns = hungarian.lap(matrix)
    interesting_indicies = []
    for i in range(0, len(df_sample_condition)):
        interesting_indicies.append(col_assigns[i])
    return df_population.ix[interesting_indicies]


def discover_weightings(df_full):
    """Returns a dict of weightings for every column in file_column_definitions
    which is not labeled as ignore, and is not the match_attribute.  This function
    requires that there are no null/None/NaN values in the columns.
    
    TODO: Remove the requirement for no null/None/NaN values.
    """
    candidates = []
    for column_name in df_columns.columns:
        if df_columns[column_name][0] != "ignore":
            if column_name != match_attribute:
                candidates.append(column_name)

    clf = linear_model.LinearRegression()

    r_column = df_full[match_attribute]
    nr_columns = df_full[candidates]
    clf.fit(nr_columns, r_column)
    results = {}
    for i in range(0, len(clf.coef_)):
        results[candidates[i]] = clf.coef_[i]
    return results


def diff_two_rows(x, y):
    """Returns difference over all columns in column_definitions between two rows
    in a pandas dataframe as a tuple: (average difference, dictionary of column differences)
    """
    diffs = {}
    for column_name in df_columns.columns:
        difference = diff(x[column_name], y[column_name], column_name)
        if difference != None:
            diffs[column_name] = difference
    return (np.mean(diffs.values()), diffs)


def diff_ignore(one, two):
    return None


def diff_nominal(one, two):
    if one == two:
        return 0
    else:
        return 1


def diff_ordinal(one, two, sorted_range):
    #sorted_range=sorted(sorted_range)
    pos1 = sorted_range.index(one)
    pos2 = sorted_range.index(two)
    diff = math.fabs(pos1 - pos2)
    return diff / (len(sorted_range) - 1)


def diff_real(one, two, min, max):
    top = float(max - min)
    one = float(one)
    two - float(two)
    return math.fabs((one / top) - (two / top))


def diff(one, two, column_name):
    #look up the type for the column name
    return column_funs[column_name](one, two, **column_parameters[column_name])


def load_data():
    global df_full
    global df_columns

    df_full = pd.read_csv(file_input, na_values=["", " ", "NULL"])
    df_columns = pd.read_csv(file_columns, na_values=["", " ", "NULL"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Subsamples a large population based on the characteristics of a small population.')
    parser.add_argument('-i', '--input', help='Input filename as a CSV')
    parser.add_argument('-o', '--output', help='Output filename for matches')
    parser.add_argument('-c', '--column_definitions', help='The CSV of column definitions')
    parser.add_argument('-m', '--match', help='The attribute to match one')
    parser.add_argument('-w', '--weights', help='Whether weights should be automatically discovered or evenly applied')
    args = parser.parse_args()

    file_columns = args.column_definitions
    file_input = args.input
    file_output = args.output
    match_attribute = args.match
    weights = args.weights

    #load datafiles into pandas dataframes
    load_data()

    #separate the general population for matching and the subsample of interest based on the match_attribute
    df_population = df_full[df_full[match_attribute] == 0]
    df_sample_condition = df_full[df_full[match_attribute] == 1]
    df_population = df_population.reset_index(drop=True)
    df_sample_condition = df_sample_condition.reset_index(drop=True)

    #make sure sample condition and population and column definitions are all the same size
    assert len(df_population.columns) == len(df_columns.columns) == len(
        df_sample_condition.columns), "All data files must have the same number of columns"
    #make sure that population file has at least one free choice in it
    assert len(df_population) > len(
        df_sample_condition), "The population file must have more items in it than the sample condition file"

    column_weights = {}
    if weights == "auto":
        #run logistic regression to discover the weightings for each column
        column_weights = discover_weightings(df_full)
    else:
        #set all columns to 1 except those that will not be used
        for column_name in df_columns.columns:
            if df_columns[column_name][0] != "ignore":
                column_weights[column_name] = 1.0
            else:
                column_weights[column_name] = 0.0

    #go through and build shorthand variables for the columns based on the data in the definitions file
    for column_name in df_columns:
        if df_columns[column_name][0] == "ordinal":
            items = []
            items.extend(df_sample_condition[column_name].unique().tolist())
            items.extend(df_population[column_name].unique().tolist())
            s = set()
            for item in items:
                s.add(item)
            #print "Setting ordinal " + str(column_name)
            newlist = sorted(s)
            column_parameters[column_name] = {"sorted_range": newlist}
            column_funs[column_name] = diff_ordinal
        elif df_columns[column_name][0] == "real":
            mymin = np.min(df_sample_condition[column_name])
            if np.min(df_population[column_name]) < mymin:
                mymin = np.min(df_population[column_name])
            mymax = np.max(df_sample_condition[column_name])
            if np.max(df_population[column_name]) > mymax:
                mymax = np.max(df_population[column_name])
            #print "Setting real " + str(column_name)
            column_parameters[column_name] = {"min": mymin, "max": mymax}
            column_funs[column_name] = diff_real
        elif df_columns[column_name][0] == "ignore":
            #print "Setting unique " + str(column_name)
            column_parameters[column_name] = {}
            column_funs[column_name] = diff_ignore
        elif df_columns[column_name][0] == "nominal":
            #print "Setting nominal " + str(column_name)
            column_parameters[column_name] = {}
            column_funs[column_name] = diff_nominal
        #add a zero index for anything that does not already have a weight
        if column_name not in column_weights.keys():
            column_weights[column_name] = 0

    #create a matrix filled with ones (worst match value)
    matrix = np.ones((len(df_population), len(df_population)), dtype=np.float32)

    print "Building difference tables matrix of size (" + str(len(df_population)) + "x" + str(
        len(df_population)) + "):",
    x_i = 0
    for x in df_population.iterrows():
        y_i = 0
        for y in df_sample_condition.iterrows():
            diffs = []
            for column_name in df_columns.columns:
                difference = diff(x[1][column_name], y[1][column_name], column_name)
                #todo: right now if a value is missing we maximize it, setting it to totally different at 1, is this reasonable?
                #todo: instead should we just ignore this?  or should it be a sort of special value?
                if np.isnan(difference):
                    difference = 1
                if difference != None:
                    diffs.append(difference * column_weights[column_name])
            matrix[x_i][y_i] = np.sum(diffs)
            y_i += 1
        x_i += 1
        print ".",
        sys.stdout.flush()
    print""

    #run lap solver
    print "Running the LAP using the hungarian method."
    df_matches = run_hungarian(matrix, df_population, df_sample_condition)

    df_matches = df_matches.reset_index(drop=True)

    #write output files
    write_output(file_output, df_sample_condition, df_matches)

    #write output report
    print_report()

    """
    
    #Alternative matches
    import time
    before = int(round(time.time() * 1000))
    print before
    
    matches={}
    for rowiter in df_deltas.iterrows():
        diff_value=np.mean(rowiter[1])
        curr_sample_val=rowiter[0]
        candidates=[]
        for popiter in df_population.iterrows():
            difference_value=diff_two_rows(df_sample_condition.ix[curr_sample_val], df_population.ix[popiter[0]])[0]
            if difference_value <= diff_value:
                candidates.append( (popiter[0], difference_value) )
            matches[curr_sample_val]=candidates
            
    after = int(round(time.time() * 1000))
    print after
    print str(after-before)
    """



    #non_regression=df_john[["GENDER_CODE","BIRTH_YEAR","ABORIGINAL_ANCESTRY_IND","SELF_REPORTED_DISABILITY_IND","cu_year1","entrance_average"]]
    #clf.fit(non_regression,regression_val)
    #clf.coef_

