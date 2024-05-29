import math
import random

def evaluate(features):
    # Initialize your data loader, classifier, and validator here...
    data_loader = DataLoader("small-test-dataset.txt")
    classifier = Classifier()
    validator = Validator(classifier)

    # Load the data and labels
    data = []
    labels = []
    for label, instances in data_loader.dataVals.items():
        data.extend(instances)
        labels.extend([label] * len(instances))

    return validator.validate(data, labels, features)


def forward_selection(features):
    running_features = []
    remaining = features.copy()

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


def backward_elimination(features):
    current_set = set(features.copy())
    best_featureset = current_set.copy()
    best_setscore = evaluate(current_set) * 100  # Evaluate the full set
    print(f"Using feature(s) {set(current_set)} accuracy is {best_setscore:.1f}%")

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
        if worst_feature and highest_score > best_setscore:
            current_set.remove(worst_feature)
            best_setscore = highest_score
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

        # Normalize the data
        # self.normalizeData()

    # def normalizeData(self):
    #     for classVal, instances in self.dataVals.items():
    #         # Calculate mean and standard deviation for each feature
    #         num_features = len(instances[0])
    #         normalized_instances = []
    #         for instance in instances:
    #             normalized_instance = []
    #             for i in range(num_features):
    #                 feature_values = [instance[i] for instance in instances]
    #                 mean = sum(feature_values) / len(feature_values)
    #                 variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
    #                 std_dev = variance ** 0.5

    #                 # Normalize feature values
    #                 normalized_instance.append((instance[i] - mean) / std_dev)
    #             normalized_instances.append(normalized_instance)
    #         self.dataVals[classVal] = normalized_instances

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