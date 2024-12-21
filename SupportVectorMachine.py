import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import time
import dataHandling as dh

def SVM(Kernel, verbal):
    input_data, input_labels, test_data, test_labels = dh.load_data("DB",0)

    input_data = input_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0

    # Apply PCA to reduce dimensions to 50 components
    x_train_pca, x_test_pca = dh.apply_pca(input_data, test_data, n_components=200)

    # Initialize an SVM classifier
    start_time=time.time()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_pca)
    x_test = scaler.transform(x_test_pca)

    # Train an SVM classifier
    if (Kernel=='poly'):
        svm_model = SVC(kernel='poly', degree=4, coef0=1, C=1.0, verbose=verbal)
    elif (Kernel=='linear'):
        svm_model = SVC(kernel='linear', C=0.1, verbose=verbal)
    elif (Kernel=='rbf'):
        svm_model = SVC(kernel='rbf', C=1.0, gamma='auto', verbose=verbal)
    elif (Kernel=='sigmoid'):
        svm_model = SVC(kernel='sigmoid', C=10.0, coef0=0, tol=1e-4, verbose=verbal)
    else:
        print("Wrong Kernel value upon calling the function. \n A runtime error will stop the code execution.")
    print(f"Training the SVM model with {Kernel} ...")
    svm_model.fit(x_train, input_labels)
    total_time=time.time() - start_time
    minutes= total_time//60
    seconds= total_time - minutes*60
    print(f"SVM Training/Classification complete in {int(minutes)} minutes, {seconds:.10f} seconds")

    # Evaluate the SVM
    print("Evaluating the SVM model...")
    predicted_labels = svm_model.predict(x_test)

    # Metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels))
