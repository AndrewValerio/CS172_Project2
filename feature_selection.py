import random

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
