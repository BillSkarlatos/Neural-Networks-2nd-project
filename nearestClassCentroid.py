from sklearn.metrics import accuracy_score
import numpy as np
import dataHandling as dh
import time

# Nearest Class Centroid. 
def NCC(database):
    num_classes=10
    input_data, input_labels, test_data, test_labels = dh.load_data(database, 0)
    centroids = np.zeros((num_classes, input_data.shape[1]))

    start_time=time.time()
    for class_label in range(num_classes):
        class_data = input_data[input_labels == class_label]  # Images of this caregory/class.
        centroids[class_label] = np.mean(class_data, axis=0)  # Centroid calculation.
    total_time=time.time() - start_time
    minutes= total_time//60
    seconds= total_time - minutes*60


    # Sample categorisation.
    prediction = []
    for sample in test_data:
        distances = np.linalg.norm(centroids - sample, axis=1)  # Distance from centroid.
        prediction.append(np.argmin(distances))  # Class/category of the nearest centroid.

    # Accuracy calculation.
    accuracy = accuracy_score(prediction, test_labels)
    print(f"Classification complete in {int(minutes)} minutes, {seconds:.2f} seconds")
    print(f"Accuracy with Nearest Class Centroids: {accuracy * 100:.2f}%")
    return accuracy