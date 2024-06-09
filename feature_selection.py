import math
import random
import pandas as pd
import numpy as np
import time

def evaluate(features, data_norm, labels):
    # classifier and validator here

    classifier = Classifier()
    validator = Validator(classifier)

    return validator.validate(data_norm, labels, features)


def forward_selection(features, data_norm, labels):
    running_features = []
    remaining = features.copy()
    best_setscore = -1  # Initialize best_setscore to a very low value
    best_featureset = []

    start_time = time.time()

    print("Beginning search.")

    while remaining:
        best_feature = None
        max_score = -1

        for feature in remaining:
            if feature not in running_features:
                #print("feature is: " + str(feature))
                current = running_features + [feature]
                #print("curr feature is: " + str(current_features))
                score = evaluate(current, data_norm, labels) * 100
                print(f"Using feature(s) {set(current)} accuracy is {score:.1f}%")

                if score > max_score:
                    max_score = score
                    best_feature = feature

        if best_feature is not None:    
            running_features.append(best_feature)
            if best_setscore < max_score:
                best_setscore = max_score
                best_featureset = running_features.copy()
            remaining.remove(best_feature)
            print(f"Feature set {set(running_features)} was best, accuracy is {max_score:.1f}%")

    print(f"Finished search!! The best feature subset is {set(best_featureset)} which has an accuracy of {best_setscore:.1f}%")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.10f} seconds")
    return None

def forward_selection_pruning(features, data_norm, labels):
    running_features = []
    remaining = features.copy()
    best_setscore = -1  # Initialize best_setscore to a very low value
    best_featureset = []
    overall_max_score = -1

    start_time = time.time()

    print("Beginning search.")

    while remaining:
        best_feature = None
        max_score = -1

        for feature in remaining:
            if feature not in running_features:
                #print("feature is: " + str(feature))
                current = running_features + [feature]
                #print("curr feature is: " + str(current_features))
                score = evaluate(current, data_norm, labels) * 100
                print(f"Using feature(s) {set(current)} accuracy is {score:.1f}%")

                if score > max_score:
                    max_score = score
                    best_feature = feature

        if max_score <= overall_max_score:
            print("\nMax score not improved, stopping search.")
            break

        # if best_feature and max_score > best_setscore:
        if best_feature is not None:    
            running_features.append(best_feature)
            best_setscore = max_score
            remaining.remove(best_feature)
            overall_max_score = max_score
            best_featureset = running_features.copy()
            print(f"Feature set {set(running_features)} was best, accuracy is {max_score:.1f}%")
        #else:
        #    print(f"Feature set {set(current)} accuracy is {max_score:.1f}%, did not improve from {best_setscore:.1f}%")
        #    break

    print(f"Finished search!! The best feature subset is {set(best_featureset)} which has an accuracy of {best_setscore:.1f}%")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.10f} seconds")
    return None


def backward_elimination(features, data_norm, labels):
    current_set = set(features)
    best_featureset = current_set.copy()
    overall_best = -1
    print("Beginning search.")
    
    best_setscore = evaluate(list(current_set), data_norm, labels) * 100  # Evaluate the full set
    print(f"Using feature(s) {set(current_set)} accuracy is {best_setscore:.1f}%")

    while current_set:
        highest_score = -1
        worst_feature = None
        for feature in current_set:
            temp_set = current_set.copy()
            temp_set.remove(feature)
            score = evaluate(list(temp_set), data_norm, labels) * 100
            print(f"Using feature(s) {set(temp_set)} accuracy is {score:.1f}%")
            if score > highest_score:
                worst_feature = feature
                highest_score = score
        if worst_feature and highest_score > best_setscore:
            current_set.remove(worst_feature)
            best_setscore = highest_score   
            best_featureset = current_set.copy()
        else:
            break
        print(f"Feature set {set(current_set)} was best, accuracy is {highest_score:.1f}%")   
    print(f"Finished search!! The best feature subset is {set(best_featureset)} which has an accuracy of {best_setscore:.1f}%")

    return



class Classifier:
    def train(self, data_norm, labels):
        self.training_data = data_norm
        self.training_labels = labels

    def test(self, data_norm):
        data_norm = np.array(data_norm)  # Ensure data_norm is a NumPy array
        distances = np.sqrt(np.sum((self.training_data - data_norm) ** 2, axis=1))
        predicted_label_index = np.argmin(distances)
        return self.training_labels.loc[predicted_label_index]

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def validate(self, data_norm, labels, features):
        correct_predictions = 0
        indices = np.arange(len(data_norm))
        for i in range(len(data_norm)):
            train_indices = indices[indices != i]
            train_data = data_norm.iloc[train_indices][features].reset_index(drop=True)
            train_labels = labels.iloc[train_indices].reset_index(drop=True)
            test_data = data_norm.loc[i, features]
            test_label = labels[i]

            self.classifier.train(train_data, train_labels)
            prediction = self.classifier.test(test_data)

            if prediction == test_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(data_norm)
        return accuracy