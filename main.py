import feature_selection

def main():
    print("Welcome to Your Name's Feature Selection Algorithm.")
    #total_features = int(input("Please enter total number of features: "))

    # Ask the user for the specific features they want to use
    features = input("Please enter the features you want to use, separated by commas: ")
    # Convert the features from string to list of integers
    features = list(map(int, features.split(',')))

    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Your Special Algorithm.")
    choice = int(input())

    if choice == 1:
        #feature_selection.evaluate(features)
        feature_selection.forward_selection(features)
    elif choice == 2:
        #feature_selection.evaluate(features)
        feature_selection.backward_elimination(features)
    else:
        print("Invalid choice. Please enter 1 for Forward Selection or 2 for Backward Elimination.")
        return

if __name__ == "__main__":
    main()
