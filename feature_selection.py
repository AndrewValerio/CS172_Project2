import random
import math
import numpy as np

def evaluate(features):
    # Replace this with part 2 stuff
    return random.random()

def forward_selection(total_features):
    return


def backward_elimination(total_features):
    return 

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
    
    def test(self, test_instance):
        min_distance = float('inf')
        nearest_label = None
        
        for i, train_instance in enumerate(self.training_data):
            distance = np.linalg.norm(np.array(test_instance) - np.array(train_instance))
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