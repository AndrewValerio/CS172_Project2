import random
import main

def evaluate(features):
    # Replace this with part 2 stuff
    return random.uniform(0, 1)

def forward_selection(total_features):
    return


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
    print(f"Feature set {set(best_featureset)} was best, accuracy is {best_setscore:.1f}%")

    return 
