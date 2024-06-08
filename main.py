import feature_selection
import pandas as pd

file_name = 'large-test-dataset.txt'
df = pd.read_csv(file_name,sep=r'\s+',header=None)

def main():
    print("Welcome to Your Name's Feature Selection Algorithm.")
    #total_features = int(input("Please enter total number of features: "))

    # Ask the user for the specific features they want to use
    #features = input("Please enter the features you want to use, separated by commas: ")
    # Convert the features from string to list of integers
    #features = list(map(int, features.split(',')))

    # Load the data
    file_name = 'small-test-dataset.txt'
    df = pd.read_csv(file_name,sep=r'\s+',header=None)
    labels = df.iloc[:,0]
    non_norm_data = df.iloc[:,1:]

    # Drop the features not in the user's selected feature subset
    non_norm_data = non_norm_data.drop(columns=[f for f in non_norm_data.columns if f not in features])

    # Get the list of feature column indices
    features = non_norm_data.columns.tolist()
    print(f"Features: {features}")

    # Get the total number of features (columns)
    total_features = non_norm_data.shape[1]
    print(f"Total number of features: {total_features}")

    # Normalize the data
    means = non_norm_data.mean()
    std = non_norm_data.std()
    data_norm = (non_norm_data - means)/std

    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Forward Selection with Pruning")
    choice = int(input())

    if choice == 1:
        #feature_selection.evaluate(features)
        feature_selection.forward_selection(features, data_norm, labels)
    elif choice == 2:
        #feature_selection.evaluate(features)
        feature_selection.backward_elimination(features, data_norm, labels)
    elif choice == 3:
        #feature_selection.evaluate(features)
        feature_selection.forward_selection_pruning(features, data_norm, labels)
    else:
        print("Invalid choice. Please enter 1 for Forward Selection or 2 for Backward Elimination.")
        return

if __name__ == "__main__":
    main()
