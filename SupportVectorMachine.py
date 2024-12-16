import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import dataHandling as dh

input_data, input_labels, test_data, test_labels = dh.load_data("DB",0)

input_data = input_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Apply PCA to reduce dimensions to 50 components
x_train_pca, x_test_pca = dh.apply_pca(input_data, test_data, n_components=50)

# Initialize an SVM classifier
# Optional: Use StandardScaler for better scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_pca)
x_test = scaler.transform(x_test_pca)

# Train an SVM classifier
svm_model = SVC(kernel='linear', C=1.0, verbose=True)  # Linear kernel
print("Training the SVM model...")
svm_model.fit(x_train, input_labels)


# Evaluate the SVM
print("Evaluating the SVM model...")
predicted_labels = svm_model.predict(test_data)

# Metrics
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels))