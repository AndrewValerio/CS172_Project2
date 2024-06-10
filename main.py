import feature_selection
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_features(data_norm, labels, feature1, feature2):
    # Create a scatter plot of the two features
    plt.figure(figsize=(8, 6))
    
    # Create a color map
    colors = ['red' if label == 1 else 'blue' for label in labels]
    
    plt.scatter(data_norm[feature1], data_norm[feature2], c=colors)
    plt.xlabel(f'Feature {feature1}')
    plt.ylabel(f'Feature {feature2}')
    plt.title(f'Feature {feature1} vs Feature {feature2}')
    
    # Create a legend
    red_patch = mpatches.Patch(color='red', label= f'Class 1')
    blue_patch = mpatches.Patch(color='blue', label= f'Class 2')
    plt.legend(handles=[red_patch, blue_patch])
    
    plt.show()

def main():
    print("Welcome to Group 33’s Feature Selection Algorithm.")
    file_name = str(input("Type in the name of the file to test : "))
    #total_features = int(input("Please enter total number of features: "))

    # Ask the user for the specific features they want to use
    #features = input("Please enter the features you want to use, separated by commas: ")
    # Convert the features from string to list of integers
    #features = list(map(int, features.split(',')))

    # Load the data
    #file_name = 'small-test-dataset.txt'
    df = pd.read_csv(file_name,sep=r'\s+',header=None)
    labels = df.iloc[:,0]
    non_norm_data = df.iloc[:,1:]

    # Drop the features not in the user's selected feature subset
    #non_norm_data = non_norm_data.drop(columns=[f for f in non_norm_data.columns if f not in features])

    # Get the list of feature column indices
    features = non_norm_data.columns.tolist()
    print(f"Features: {features}")

    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Forward Selection with Pruning")
    choice = int(input())

    # Get the total number of features (columns)
    print(f"This dataset has {non_norm_data.shape[1]} features (not including the class attribute), with {non_norm_data.shape[0]} instances.")

    print("Please wait while I normalize the data... Done!")
    # Normalize the data
    means = non_norm_data.mean()
    std = non_norm_data.std()
    data_norm = (non_norm_data - means)/std


    # data_norm = non_norm_data

    #print(data_norm.head())


    no_features = []
    no_feature_accuracy = feature_selection.evaluate(no_features, data_norm, labels)

    print("Running nearest neighbor with no features (default rate), using “leaving-one-out” evaluation, I get an accuracy of ", (no_feature_accuracy * 100), "%")

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
    # Call the function with two features
    plot_features(data_norm, labels, 2, 3)

if __name__ == "__main__":
    main()
