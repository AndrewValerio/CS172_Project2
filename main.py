import feature_selection

def main():
    print("Welcome to Your Name's Feature Selection Algorithm.")
    total_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Your Special Algorithm.")
    choice = int(input())
    
    if choice == 1:
        print("Using no features and 'random' evaluation, I get an accuracy of ", round(feature_selection.evaluate([]), 4) * 100, "%", sep = '')
        print("Beginning search.")
        best_features = feature_selection.forward_selection(total_features)
    elif choice == 2:
        print("Using all features and 'random' evaluation, I get an accuracy of ", round(feature_selection.evaluate(list(range(total_features))), 4) * 100, "%", sep = '')
        print("Beginning search.")
        best_features = feature_selection.backward_elimination(total_features)
    else:
        print("Invalid choice. Please enter 1 for Forward Selection or 2 for Backward Elimination.")
        return

    print("Finished search!! The best feature subset is ", best_features, " which has an accuracy of ", round(feature_selection.evaluate(best_features), 4) * 100, "%", sep = '')

if __name__ == "__main__":
    main()
