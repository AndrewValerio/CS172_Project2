import math
import random

def evaluate(features):
    # Replace this with part 2 stuff
    return random.random()

def forward_selection(total_features):
    running_features = []
    remaining = list(range(1, total_features + 1))

    print("Beginning search.")

    while remaining:
        best_feature = None
        max_score = -1

        for feature in remaining:
            #print("feature is: " + str(feature))
            current = running_features + [feature]
            #print("curr feature is: " + str(current_features))
            score = evaluate(current) * 100
            print(f"Using feature(s) {set(current)} accuracy is {score:.1f}%")

            if score > max_score:
                max_score = score
                best_feature = feature

        if best_feature is not None:
            running_features.append(best_feature)
            remaining.remove(best_feature)
            print(f"Feature set {set(running_features)} was best, accuracy is {max_score:.1f}%")
    print("\nFinal features:", running_features)
    return None


def backward_elimination(total_features):
    current_set = set(range(1, total_features + 1))
    best_featureset = current_set.copy()
    best_setscore = -1
    while current_set:
        highest_score = -1
        worst_feature = None
        for feature in current_set:
            temp_set = current_set - {feature}
            score = evaluate(temp_set) * 100
            print(f"Using feature(s) {set(temp_set)} accuracy is {score:.1f}%")
            if score > highest_score:
                worst_feature = feature
                highest_score = score
        if worst_feature:
            current_set.remove(worst_feature)
            if score > best_setscore:
                best_setscore = score
                best_featureset = current_set.copy()
        else:
            break
        print(f"Feature set {set(current_set)} was best, accuracy is {highest_score:.1f}%")   
    print(f"Finished search!! The best feature subset is {set(best_featureset)} which has an accuracy of {best_setscore:.1f}%")

    return best_featureset, best_setscore



class DataLoader:
    def __init__(self, filename):
        self.dataVals = {} # dictionary that will hold the data
        self.loadData(filename)

    def loadData(self, filename):
        self.dataVals = {} # dictionary that will hold the data
        file = open(filename, 'r')
        data = file.readlines() # read all lines into a list
        for row in data: # parse the row
            row = row.strip().split()  # remove leading/trailing whitespace and split by whitespace

            if not row:  # skip empty lines
                continue

            try:
                classVal = int(float(row[0]))  # get the instance class
            except ValueError:
                print(f"Unexpected data format: {row}")
                continue

            instances = self.dataVals.get(classVal, [])
            instances.append([float(i) for i in row[1:]])  # convert features to float and append
            self.dataVals[classVal] = instances

        file.close()

class Classifier:
    def train(self, data, labels):
        self.training_data = data
        self.training_labels = labels

    def test(self, data):
        distances = [math.sqrt(sum((x - y) ** 2 for x, y in zip(train, data))) for train in self.training_data]
        nearest_index = distances.index(min(distances))
        return self.training_labels[nearest_index]

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def validate(self, data, labels, features):
        correct_predictions = 0
        for i in range(len(data)):
            train_data = [d for j, d in enumerate(data) if j != i]
            train_data = [[d[f] for f in features] for d in train_data]
            train_labels = [l for j, l in enumerate(labels) if j != i]
            test_data = [data[i][f] for f in features]
            test_label = labels[i]

            self.classifier.train(train_data, train_labels)
            prediction = self.classifier.test(test_data)

            if prediction == test_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(data)
        return accuracy