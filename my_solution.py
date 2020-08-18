import sys
import os
import warnings
import copy
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Disable tensorflow warning: `tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA`. See https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

printProgress = True
# printProgress = False

class bcolors:
    VIOLET = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def green(string):
    return addColor(string, bcolors.OKGREEN)

def blue(string):
    return addColor(string, bcolors.OKBLUE)

def bold(string):
    return addColor(string, bcolors.BOLD)

def addColor(string, color):
    return f'{color}{string}{bcolors.ENDC}'

def log(string):
    if printProgress:
        print(string)

def find_best_solution(train_filename):
    unscaled_dist = extract_dist(train_filename)
    # unscaled_dist.info()

    unscaled_dist = clean_distribution(unscaled_dist)
    # unscaled_dist.info()
    # standardised_csv.to_csv('scaledData.csv', index=False)

    preprocessors = {
        'StandardScaler': StandardScaler(),
        # The below remove anomalies
        'RobustScaler': RobustScaler(),
        'PowerTransformer(method="yeo-johnson")': PowerTransformer(method='yeo-johnson'),
        'QuantileTransformer(output_distribution="normal")': QuantileTransformer(output_distribution="normal"),
        'QuantileTransformer(output_distribution="uniform")': QuantileTransformer(output_distribution="uniform"),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'Normalizer': Normalizer(),
    }
    distributions = {
        'unscaled': unscaled_dist,
        'StandardScaler': scale_distribution(unscaled_dist, StandardScaler()),
        'RobustScaler': scale_distribution(unscaled_dist, RobustScaler()),
        'PowerTransformer(method="yeo-johnson")': scale_distribution(unscaled_dist, PowerTransformer(method='yeo-johnson')),
        'QuantileTransformer(output_distribution="normal")': scale_distribution(unscaled_dist, QuantileTransformer(output_distribution="normal")),
        'QuantileTransformer(output_distribution="uniform")': scale_distribution(unscaled_dist, QuantileTransformer(output_distribution="uniform")),
        'MinMaxScaler': scale_distribution(unscaled_dist, MinMaxScaler()),
        'MaxAbsScaler': scale_distribution(unscaled_dist, MaxAbsScaler()),
        'Normalizer': scale_distribution(unscaled_dist, Normalizer()),
    }

    classifiers = {
        # Standard classifiers
        'KNeighborsClassifier': {
            'configurable': lambda configuration: KNeighborsClassifier(n_neighbors=configuration),
            'score_function': get_classifier_score_knn,
        },
        'LinearSVC': {
            'configurable': lambda configuration: LinearSVC(max_iter=configuration),
            'score_function': get_classifier_score_linear_svc,
        },
        'LogisticRegression': {
            'configurable': lambda configuration: LogisticRegression(solver=configuration),
            'score_function': get_classifier_score_logistic_regression,
        },
        # DL
        # 'Sequential': {
        #     # 'configurable': # TODO get this working
        #     'score_function': get_model_score_sequential,
        # },
    }

    test_sizes = [0.2, 0.25, 0.3]
    # test_sizes = [0.2]

    results_of_all_configurations = []

    log(blue(f'\n############# Trying Configurations #############'))
    for dist_name, dist in distributions.items():
        log(bold(f'\n{dist_name}'))
        X, y = get_x_matrix_and_y_vector(dist) # set up X matrix and y vectors for the test and training sets
        for test_size in test_sizes:
            log(f'    test_size = {test_size}')
            train_test_data = (X, y, test_size)
            for classifier_name, classifier in classifiers.items():
                log(f'        {classifier_name}')
                configuration_results, classifier_average_score = try_configuration(train_test_data, classifier)
                for result in configuration_results:
                    result['preprocessor_name'] = dist_name
                    result['test_size'] = test_size
                    result['classifier_name'] = classifier_name
                results_of_all_configurations.extend(configuration_results)

    results_of_all_configurations.sort(
        key=lambda result: result['configuration_score'],
        reverse=True,
    )
    logTopConfigurations(results_of_all_configurations, 10)

    return extract_best_solution(preprocessors, classifiers, results_of_all_configurations)

def logTopConfigurations(results_of_all_configurations, numTopConfigurations):
    log(blue(f'\n############# Top {numTopConfigurations} Configurations #############'))
    if numTopConfigurations > len(results_of_all_configurations):
        numTopConfigurations = len(results_of_all_configurations)
    for i in range(numTopConfigurations):
        result = results_of_all_configurations[i]
        preprocessor_name = result['preprocessor_name']
        test_size = result['test_size']
        classifier_name = result['classifier_name']
        configuration_name = result['configuration_name']
        configuration_value = result['configuration_value']
        configuration_score = result['configuration_score']
        log(f'{green(f"{configuration_score:.2f}")}: {bold(classifier_name)} classifier with {bold(f"{configuration_value} {configuration_name}")}, fitted to a distribution pre-processed with a {bold(preprocessor_name)}, and with test size {bold(test_size)}')

def extract_best_solution(preprocessors, classifiers, results_of_all_configurations):
    best_configuration = results_of_all_configurations[0]
    best_preprocessor = preprocessors[best_configuration['preprocessor_name']]
    best_classifier = best_configuration['classifier_name']
    best_configurable_classifier = classifiers[best_classifier]['configurable']
    best_configuration_value = best_configuration['configuration_value']
    best_configured_classifier = best_configurable_classifier(best_configuration_value)

    log(blue('\n############# Best Configuration #############'))
    log (f'Preprocessor: {best_preprocessor}')
    log (f'Classifier: {best_configured_classifier}')
    return best_preprocessor, best_configured_classifier


def try_configuration(train_test_data, classifier):
    # num_tests_per_configuration = 10
    num_tests_per_configuration = 1
    configuration_results, classifier_average_score = classifier['score_function'](
        train_test_data,
        num_tests_per_configuration,
    )
    return configuration_results, classifier_average_score

def get_standard_layers_for_sequential_model():
    return [
        # TODO: improve this
        tf.keras.layers.Flatten(),  # takes our 28x28 and makes it 1x784
        tf.keras.layers.Dense(128, activation=tf.nn.relu),  # a simple fully-connected layer, 128 units, relu activation
        tf.keras.layers.Dense(128, activation=tf.nn.relu),  # a simple fully-connected layer, 128 units, relu activation
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),  # our output layer. 10 units for 10 classes. Softmax for probability distribution
    ]

def get_model_score_sequential(train_test_data, num_tests_per_configuration):
    results = []
    # configuration_average_score = 0
    configurations = {
        'standard_layers': get_standard_layers_for_sequential_model(),
    }
    for configuration_name, layers in configurations.items():
        model_average_loss = 0
        model_average_accuracy = 0
        for i in range(0, num_tests_per_configuration):
            # TODO: why is the train time increasing?
            model = tf.keras.models.Sequential()  # a basic feed-forward model
            for layer in layers:
                model.add(layer)
            model.compile(
                optimizer='adam',  # Good default optimizer to start with
                loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
                metrics=['accuracy'],  # what to track
            )

            (X, y, test_size) = train_test_data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X_train.values, y_train.values, epochs=4, verbose=0)
            loss, accuracy = calculate_model_score(model, X_test, y_test)
            model_average_loss += loss
            model_average_accuracy += accuracy
        model_average_loss /= num_tests_per_configuration
        model_average_accuracy /= num_tests_per_configuration
        # model.summary()
        log(f'            layers = {configuration_name}, loss = {model_average_loss:.2f}, accuracy = {model_average_accuracy:.2f}')
        # model.save(f'./models/{dist_name}.model')
        # configuration_average_score += model_average_accuracy
        results.append({
            'configuration_name': 'layers',
            'configuration_value': configuration_name, # using this because `layers` does not print nicely, and we're not actually using this model to output results
            'configuration_score': model_average_accuracy # TODO should I care about accuracy or loss?
        })
    # configuration_average_score /= len(configurations)

    # temp_model.summary()
    return results, model_average_accuracy

def get_classifier_score_linear_svc(train_test_data, num_tests_per_configuration):
    results = []
    classifier_average_score = 0
    # configurations = [10000]
    configurations = range(5000, 15001, 5000)
    for configuration in configurations:
        configured_classifier = LinearSVC(max_iter=configuration)
        configuration_score = get_average_configuration_score(num_tests_per_configuration, configured_classifier, train_test_data)
        log(f'            max_iter = {configuration}: {configuration_score:.2f}')
        classifier_average_score += configuration_score
        results.append({
            'configuration_name': 'max_iter',
            'configuration_value': configuration,
            'configuration_score': configuration_score
        })
    classifier_average_score /= len(configurations)
    return results, classifier_average_score

def get_classifier_score_logistic_regression(train_test_data, num_tests_per_configuration):
    results = []
    classifier_average_score = 0
    configurations = ['liblinear']
    for configuration in configurations:
        configured_classifier = LogisticRegression(solver=configuration)
        configuration_score = get_average_configuration_score(num_tests_per_configuration, configured_classifier, train_test_data)
        log(f'            solver = {configuration}: {configuration_score:.2f}')
        classifier_average_score += configuration_score
        results.append({
            'configuration_name': 'solver',
            'configuration_value': configuration,
            'configuration_score': configuration_score
        })
    classifier_average_score /= len(configurations)
    return results, classifier_average_score

def get_classifier_score_knn(train_test_data, num_tests_per_configuration):
    results = []
    classifier_average_score = 0
    configurations = range(3, 30, 4)
    # configurations = [22]
    for configuration in configurations:
        configured_classifier = KNeighborsClassifier(n_neighbors=configuration)
        configuration_score = get_average_configuration_score(num_tests_per_configuration, configured_classifier, train_test_data)
        log(f'            k = {configuration}: {configuration_score:.2f}')
        classifier_average_score += configuration_score
        results.append({
            'configuration_name': 'n_neighbors',
            'configuration_value': configuration,
            'configuration_score': configuration_score
        })
    classifier_average_score /= len(configurations)
    return results, classifier_average_score

@ignore_warnings(category=ConvergenceWarning)  # stop this warning polluting our prediction output
def get_average_configuration_score(num_tests, configured_classifier, train_test_data):
    temp_classifier = copy.deepcopy(configured_classifier)
    average_configuration_score = 0
    for i in range(0, num_tests):
        (X, y, test_size) = train_test_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        temp_classifier.fit(X_train, y_train)
        average_configuration_score += calculate_classifier_score(temp_classifier, X_test, y_test)
    average_configuration_score /= num_tests
    return average_configuration_score

def calculate_model_score(model, X_test, y_test):
    # TODO: why isn't this outputting 1s and 0s?
    # Do I need to StandardScaler.inverse_transform()?
    predictions = model.predict(X_test)
    # for prediction in predictions:
    #     print(prediction)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)  # evaluate the out of sample data with model
    return loss, accuracy

def calculate_classifier_score(classifier, X_test, y_test):
    # predictions = classifier.predict(X_test)
    # for prediction in predictions:
    #     print(prediction)
    score = classifier.score(X_test, y_test)
    return score

def extract_dist(csv_filename):
    csv = pd.read_csv(csv_filename)
    return csv

def clean_distribution(original_dist):
    dist = map_non_numerical_values_to_numbers(original_dist)
    dist = handle_nan_values(dist)
    return dist

def map_non_numerical_values_to_numbers(original_dist):
    dist = copy.deepcopy(original_dist)

    dist.sex.replace(to_replace='female', value=0, inplace=True)
    dist.sex.replace(to_replace='male', value=1, inplace=True)

    # map the ports of embarkation to numbers
    # C = Cherbourg; Q = Queenstown; S = Southampton
    dist.embarked.replace(to_replace='C', value=0, inplace=True)
    dist.embarked.replace(to_replace='Q', value=1, inplace=True)
    dist.embarked.replace(to_replace='S', value=2, inplace=True)

    return dist

def handle_nan_values(original_dist):
    dist = copy.deepcopy(original_dist)

    dist.age.fillna(dist.age.median(), inplace=True)
    # dist.age.fillna(-1, inplace=True)
    dist.embarked.fillna(-1, inplace=True)
    dist.fare.fillna(-1, inplace=True)

    return dist

def get_x_matrix_and_y_vector(dist):
    # set up our feature matrix
    X = pd.concat([
        dist.pclass,
        dist.sex,
        dist.age,
        dist.sibsp,
        dist.parch,
        dist.fare,
        dist.embarked,
        # Ignore other columns:
        # dist.name,
        # dist.ticket,
        # dist.cabin,
    ], axis=1)

    # set up our vector of labels
    try:
        y = dist.survived
    except AttributeError:
        y = None

    return X, y

def scale_distribution(original_dist, scaler):
    dist = copy.deepcopy(original_dist)
    columns_to_scale = ['sex', 'age', 'fare', 'embarked']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dist[columns_to_scale] = scaler.fit_transform(dist[columns_to_scale])
    return dist

def output_solution(
    train_filename,
    test_filename,
    preprocessor,
    configured_classifier,
    ):
    unclean_train = extract_dist(train_filename)
    unclean_test = extract_dist(test_filename)

    unscaled_train = clean_distribution(unclean_train)
    unscaled_test = clean_distribution(unclean_test)

    train = scale_distribution(unscaled_train, preprocessor)
    test = scale_distribution(unscaled_test, preprocessor)

    X_train, y_train = get_x_matrix_and_y_vector(train)
    X_test, y_test = get_x_matrix_and_y_vector(test)

    clf = configured_classifier.fit(X_train, y_train)

    log(blue(f'\n############# Predictions #############'))
    predictions = clf.predict(X_test)
    for prediction in predictions:
        print(prediction)

def output_best_solution(train_filename, test_filename):
    output_solution(
        train_filename,
        test_filename,
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=22),
    )

if __name__ == '__main__':
    try:
        already_have_best_solution = False
        train_filename = sys.argv[1]

        if len(sys.argv) == 2:
            find_best_solution(train_filename)
        else:
            test_filename = sys.argv[2]
            if already_have_best_solution:
                output_best_solution(train_filename, test_filename)
            else:
                preprocessor, configured_classifier = find_best_solution(train_filename)
                output_solution(
                    train_filename,
                    test_filename,
                    preprocessor,
                    configured_classifier,
                )

    except IndexError:
        print(f'Usage: {sys.argv[0]} TRAIN_FILE TEST_FILE')
