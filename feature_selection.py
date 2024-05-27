import random
import math
import numpy as np

def evaluate(features):
    return random.uniform(0, 1)

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

def loadData(self, filename = "small-test-dataset.txt"):
    self.dataVals = {} # dictionary that will hold the data
    file = open(filename, 'r')
    data = file.readlines() # read all lines into a list
    for row in data: # parse the row
        row = row.split('\n')
        row = row[0].split(' ')
        row.remove('')

        classVal = int(row[0][0]) # get the instance class

        for i in row[1:]:
            for j in i.split(): # takes care of any whitespace that got through
                instances = self.dataVals.get(classVal, [])
                instances.append(float(j)) # python float() converts IEEE to double automatically
                self.dataVals[classVal] = instances
    file.close()

class Classifier_Class():
    def __init__(self):
        self.training_data = []
        self.labels = []

    def train(self, data):
        for class_label, instances in data.items():
            for instance in instances:
                self.training_data.append(instance)
                self.labels.append(class_label)
    
    def test(self, testing_instance):
        min_distance = float('inf')
        nearest_label = None
        for i, train_instance in enumerate(self.training_data):
            distance = np.linalg.norm(np.array(testing_instance) - np.array(train_instance))
            if distance < min_distance:
                min_distance = distance
                nearest_label = self.labels[i]
        return nearest_label
    
    def nearest_neighbor(self, dataset, point, feature_subset, number_of_instances):
        nearest_neighbor = 0
        shortest_distance = float('inf')
        for i in range(number_of_instances):
            if point == i:
                continue
            else:
                distance = 0
                for j in feature_subset:
                    distance += pow((dataset[i][j] - dataset[point][j]), 2)
                distance = math.sqrt(distance)
                if distance < shortest_distance:
                    nearest_neighbor = i 
                    shortest_distance = distance           
        return nearest_neighbor