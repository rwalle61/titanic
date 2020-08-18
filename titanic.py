import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_X_and_y(filename):
    ''' Reads CSV values from the given file name and returns the X matrix and
        y vector '''
    # read the data
    csv = pd.read_csv(filename)

    # map all females to 0 and all males to 1
    csv.sex.replace(to_replace='female', value=0, inplace=True)
    csv.sex.replace(to_replace='male', value=1, inplace=True)

    # map all NaN ages to -1
    csv.age.fillna(-1, inplace=True)

    # map all NaN fares to -1
    csv.fare.fillna(-1, inplace=True)

    # map the ports of embarkation to numbers
    # C = Cherbourg; Q = Queenstown; S = Southampton
    csv.embarked.fillna(-1, inplace=True)
    csv.embarked.replace(to_replace='C', value=0, inplace=True)
    csv.embarked.replace(to_replace='Q', value=1, inplace=True)
    csv.embarked.replace(to_replace='S', value=2, inplace=True)

    # set up our feature matrix
    X = pd.concat([
        csv.pclass,
        csv.sex,
        csv.age,
        csv.sibsp,
        csv.parch,
        csv.fare,
        csv.embarked
    ], axis=1)

    # set up our vector of labels
    try:
        y = csv.survived
    except AttributeError:
        y = None

    return X, y


if __name__ == '__main__':
    try:
        # get files names from the command line
        train_filename = sys.argv[1]
        test_filename = sys.argv[2]

        # set up X matrix and y vectors for the test set and training set
        X_train, y_train = get_X_and_y(train_filename)
        X_test, y_test = get_X_and_y(test_filename)

        # fit the model
        clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)

        predictions = clf.predict(X_test)
        for prediction in predictions:
            print(prediction)

        # print the accuracy of the fitted model against the test set
        # commented out since this is only possible if the y_test matrix is
        # available i.e. this is useful for development only with a train.csv
        # print('%.6f' % clf.score(X_test, y_test))

    except IndexError:
        print('Usage: %s TRAIN_FILE TEST_FILE' % sys.argv[0])
