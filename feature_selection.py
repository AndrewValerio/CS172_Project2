import math
import random
import pandas as pd
import numpy as np
file_name = 'small-test-dataset.txt'
df = pd.read_csv(file_name,sep=r'\s+',header=None)

def evaluate(features):
    # classifier and validator here

    classifier = Classifier()
    validator = Validator(classifier)

    labels = df.iloc[:,0]
    non_norm_data = df.iloc[:,1:]

    means = non_norm_data.mean()
    std = non_norm_data.std()

    data_norm = (non_norm_data - means)/std

    return validator.validate(data_norm, labels, features)


def forward_selection(features):
    running_features = []
    remaining = features.copy()
    best_setscore = -1  # Initialize best_setscore to a very low value
    best_featureset = []

    print("Beginning search.")

    while remaining:
        best_feature = None
        max_score = -1

        for feature in remaining:
            if feature not in running_features:
                #print("feature is: " + str(feature))
                current = running_features + [feature]
                #print("curr feature is: " + str(current_features))
                score = evaluate(current) * 100
                print(f"Using feature(s) {set(current)} accuracy is {score:.1f}%")

                if score > max_score:
                    max_score = score
                    best_feature = feature

        if best_feature and max_score > best_setscore:
            running_features.append(best_feature)
            best_setscore = max_score
            best_featureset = running_features.copy()
            print(f"Feature set {set(running_features)} was best, accuracy is {max_score:.1f}%")
        else:
            print(f"Feature set {set(current)} accuracy is {max_score:.1f}%, did not improve from {best_setscore:.1f}%")
            break

    print(f"Finished search!! The best feature subset is {set(best_featureset)} which has an accuracy of {best_setscore:.1f}%")
    return None


def backward_elimination(features):
    current_set = features.copy()
    best_featureset = current_set.copy()
    best_setscore = evaluate(current_set) * 100  # Evaluate the full set
    print(f"Using feature(s) {set(current_set)} accuracy is {best_setscore:.1f}%")

    while current_set:
        highest_score = -1
        worst_feature = None
        for feature in current_set:
            temp_set = current_set.copy()
            temp_set.remove(feature)
            score = evaluate(temp_set) * 100
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

    return best_featureset, best_setscore



class Classifier:
    def train(self, data_norm, labels):
        self.training_data = data_norm
        self.training_labels = labels

    def test(self, data_norm):
        distances = self.training_data.apply(lambda x: np.sqrt(np.sum((x - data_norm) ** 2)), axis=1)
        predicted_label_index = np.argmin(distances)
        return self.training_labels.loc[predicted_label_index]

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def validate(self, data_norm, labels, features):
        correct_predictions = 0
        for i in range(len(data_norm)):
            train_data = data_norm.drop(i).reset_index(drop=True)
            train_labels = labels.drop(i).reset_index(drop=True)
            test_data = data_norm.loc[i, features]
            test_label = labels[i]

            self.classifier.train(train_data[features], train_labels)
            prediction = self.classifier.test(test_data)

            if prediction == test_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(data_norm)
        return accuracy