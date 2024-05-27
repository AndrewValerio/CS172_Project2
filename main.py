import feature_selection

def main():
    print("Welcome to Your Name's Feature Selection Algorithm.")
    total_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Your Special Algorithm.")
    choice = int(input())

    # Initialize your data loader, classifier, and validator here...
    data_loader = feature_selection.DataLoader("small-test-dataset.txt")
    classifier = feature_selection.Classifier()
    validator = feature_selection.Validator(classifier)

    # Load the data and labels
    data = []
    labels = []
    for label, instances in data_loader.dataVals.items():
        data.extend(instances)
        labels.extend([label] * len(instances))

    if choice == 1:
        # Modify the evaluate function to use the validator
        def evaluate(features):
            return validator.validate(data, labels, features)
        feature_selection.evaluate = evaluate
        feature_selection.forward_selection(total_features)
    elif choice == 2:
        # Modify the evaluate function to use the validator
        def evaluate(features):
            return validator.validate(data, labels, features)
        feature_selection.evaluate = evaluate
        feature_selection.backward_elimination(total_features)
    else:
        print("Invalid choice. Please enter 1 for Forward Selection or 2 for Backward Elimination.")
        return

if __name__ == "__main__":
    main()
