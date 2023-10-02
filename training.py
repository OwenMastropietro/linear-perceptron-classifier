'''
Yes
'''


import numpy as np
from numpy import ndarray
from feature_extraction import get_image_features
from helpers import extract_images, get_black_white

# pylint: disable=invalid-name


##########################################################################
#                                                                        #
# START: Build Training, Validation, and Testing Sets                    #
#                                                                        #
##########################################################################


def get_training_data(VERBOSE: bool = False) -> ndarray:
    '''
    Returns a `9990x12` set of `training_data`.

    Builds `training_data` from the files corresponding to training
    found in the 'train_and_valid' folder.

    Each row of the `training_data` corresponds to a handwritten digit.
        - row[0:9] contains the 9 `feature` values.
        - row[10] contains the `threshold` value.
        - row[11] contains the `class label`.
    '''

    if VERBOSE:
        print(f'\n{"-" * 70}')
        print('Building Training Set...')

    training_data = []
    class_labels = []

    # pylint: disable-next=unused-variable
    for i in range(NUM_FILES := 10):
        FILENAME = f'input_files/training_data/handwritten_samples_{i}.csv'
        IMAGES, LABELS = extract_images(file=FILENAME)

        for label in LABELS:
            class_labels.append([label])

        if VERBOSE:
            print(f'\tComputing Feature Values in < {FILENAME} >')

        for image in IMAGES:
            binary_image = get_black_white(image)
            training_data.append(get_image_features(binary_image))

    # Create Column 11 of threshold values = -1.
    # 9,990 instead of 10,000 because we dropped the 10 labels.
    THRESHOLDS = np.full(shape=(9990, 1), fill_value=-1)

    # Concatenate Threshhold Column to TRAIN
    training_data = np.concatenate((training_data, THRESHOLDS), axis=1)

    # Concatenate Label Column to TRAIN
    training_data = np.concatenate((training_data, class_labels), axis=1)

    np.random.shuffle(training_data)

    assert isinstance(training_data, ndarray)

    return training_data


def get_validation_data(VERBOSE: bool = False) -> ndarray:
    '''
    Returns a `2490x12` set of `validation_data`.

    Builds `validation_data` from the files corresponding to validation
    found in the 'train_and_valid' folder.
    '''

    if VERBOSE:
        print(f'\n{"-" * 70}')
        print('Building Validation Set...')

    validation_data = []
    class_labels = []

    # pylint: disable-next=unused-variable
    for i in range(NUM_FILES := 10):
        FILENAME = f'input_files/validation_data/handwritten_samples_{i}.csv'
        IMAGES, LABELS = extract_images(file=FILENAME)

        for label in LABELS:
            class_labels.append([label])

        if VERBOSE:
            print(f'\tComputing Feature Values in < {FILENAME} >')

        for image in IMAGES:
            binary_image = get_black_white(image)
            validation_data.append(get_image_features(binary_image))

    # Create Column 11 of threshold values = -1.
    # 9,990 instead of 10,000 because we dropped the 10 labels.
    thresh_arr = np.full(shape=(2490, 1), fill_value=-1)

    # Concatenate Threshhold Column to TRAIN")
    validation_data = np.concatenate((validation_data, thresh_arr), axis=1)

    # Concatenate Labels Column to TRAIN")
    validation_data = np.concatenate((validation_data, class_labels), axis=1)

    # Randomly Permuting Rows of Training Data
    np.random.shuffle(validation_data)

    assert isinstance(validation_data, ndarray)

    return validation_data


def get_testing_data(FILE: str, VERBOSE: bool = False) -> ndarray:
    '''
    Returns a `len(file) x 11` set of `testing_data`.

    Builds `testing_data` from the files corresponding to testing
    found in the 'test' folder.
    '''

    assert isinstance(FILE, str)

    if VERBOSE:
        print(f'\n{"-" * 70}')
        print('Building Testing Set...')

    testing_data = []

    IMAGES, _ = extract_images(FILE, has_label=False)

    for image in IMAGES:
        binary_image = get_black_white(image)
        testing_data.append(get_image_features(binary_image))

    THRESHOLDS = np.full(shape=(len(IMAGES), 1), fill_value=-1)

    testing_data = np.concatenate((testing_data, THRESHOLDS), axis=1)

    assert isinstance(testing_data, ndarray)

    return testing_data


##########################################################################
#                                                                        #
# END: Build Training, Validation, and Testing Sets                      #
#                                                                        #
##########################################################################


##########################################################################
#                                                                        #
# START: Train, Validate, and Test Weights                               #
#                                                                        #
##########################################################################


def train(weight_vectors: ndarray, epochs: int) -> tuple[int, int, int]:
    '''
    - Paramters: An array of 'weight_vectors' to train and the number of
    'epochs' to train the weight_vectors over.
    - Returns: An ndarray of 'weight_vectors' that have been processed / trained
    over the specified number of epochs and determined to be the most promising
    for use in classification.
    - Prediction Method: argmax {wk, xk} for all k labels / weight vectors.
    - Adjustment Method:
        - weight_vectors[j] += np.multiply(η, x[0:10])
        - weight_vectors[predict] -= np.multiply(η, x[0:10])
    - Description: This method will generate the training data and validation
    data to train the 'weight_vectors' with.
    For each image in the training data, this method will make a prediction
    using that image's features and a current set of weights to associate with
    those features. If the prediction is correct, our algorithm marks it as a
    successful prediction and moves on. If the prediction is incorrect, the
    algorithm will count it as an error and increase the weights associated with
    the actual digit's feature values while decreasing those of the incorrectly
    predicted digit.
    This will occur for each epoch up to the specified number of 'epochs',
    storing the weight_vectors after each epoch. Then, weight_vectors associated
    with the least amount of errors is selected as the 'best_weights' found and
    to be returned.
    '''

    TRAIN = get_training_data(VERBOSE=True)

    VALID = get_validation_data(VERBOSE=True)

    weights_after_each_epoch = []
    successes_for_each_epoch = []
    errors_for_each_epoch = []

    # pylint: disable-next=non-ascii-name
    η = float(0.08)  # Learning Constant, η (Eta).
    for _ in range(epochs):
        for row in TRAIN:
            class_label = int(row[10])

            features = row[0:10]
            logits = [np.dot(weights, features) for weights in weight_vectors]
            predicted_label = np.argmax(logits)

            if predicted_label != class_label:
                weight_vectors[class_label] += np.multiply(η, features)
                weight_vectors[predicted_label] -= np.multiply(η, features)

        SUCCESSES, ERRORS = validate(weight_vectors, VALID)

        weights_after_each_epoch.append(weight_vectors.copy())
        successes_for_each_epoch.append(SUCCESSES)
        errors_for_each_epoch.append(ERRORS)

    BEST_WEIGHTS_IDX = np.argmin(errors_for_each_epoch)
    BEST_WEIGHTS = weights_after_each_epoch[BEST_WEIGHTS_IDX]

    TOTAL_SUCCESSES = np.sum(successes_for_each_epoch)
    TOTAL_ERRORS = np.sum(errors_for_each_epoch)

    return BEST_WEIGHTS, TOTAL_SUCCESSES, TOTAL_ERRORS


def validate(WEIGHT_VECTORS: ndarray, VALID: ndarray) -> tuple[int, int]:
    '''
    Returns the number of successful and unsuccessful predictions made using
    the given `WEIGHT_VECTORS` on the given `VALID`ation set.
    '''

    assert isinstance(WEIGHT_VECTORS, ndarray)
    assert isinstance(VALID, ndarray)

    num_successes, num_errors = 0, 0

    for row in VALID:
        class_label = int(row[10])

        # res = []
        # for weights in WEIGHT_VECTORS:
        #     res.append(np.dot(weights, features := row[0:10]))
        features = row[0:10]
        logits = [np.dot(weights, features) for weights in WEIGHT_VECTORS]

        predicted_label = np.argmax(logits)

        if predicted_label == class_label:
            num_successes += 1
        else:
            num_errors += 1

    return num_successes, num_errors


def get_predictions(FILE: str, WEIGHT_VECTORS: ndarray) -> list:
    '''
    Returns `predictions` for each image / handwritten digit in
    the given `FILE` using the given `WEIGHT_VECTORS`.
    '''

    assert isinstance(FILE, str)
    assert isinstance(WEIGHT_VECTORS, ndarray)

    predictions = []

    TEST = get_testing_data(FILE, VERBOSE=True)

    for row in TEST:
        # I'm not sure if logits is the right term here.
        # logits + softmax = prediction ?
        features = row[0:10]
        logits = [np.dot(weights, features) for weights in WEIGHT_VECTORS]
        predictions.append(prediction := np.argmax(logits))
        # less readable version:
        # predictions.append(np.argmax(np.dot(WEIGHT_VECTORS, features)))

    assert isinstance(predictions, list)
    assert all(isinstance(prediction, np.int64) for prediction in predictions)

    return predictions


##########################################################################
#                                                                        #
# END: Train, Validate, and Test Weights                                 #
#                                                                        #
##########################################################################
