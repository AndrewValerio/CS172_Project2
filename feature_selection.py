import random

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

