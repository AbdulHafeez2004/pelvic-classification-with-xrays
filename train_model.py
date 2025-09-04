import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocessing import load_images_from_directory, extract_features

def train_model():
    print("Loading images...")

    # Update this path to your dataset location
    dataset_path = 'new_dataset'

    X, y = load_images_from_directory(dataset_path)
    
    print(f"Loaded {len(X)} images")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")
    
    print("Extracting features...")
    X_features = extract_features(X)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print("Training model...")
    # Create pipeline with PCA and Random Forest
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, verbose=1))
    ])
    
    pipeline.fit(X_train, y_train_encoded)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save model
    model_data = {
        'model': pipeline,
        'label_encoder': le,
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, 'gender_classifier.pkl')
    print("Model saved as 'gender_classifier.pkl'")
    
    return pipeline, le, accuracy

if __name__ == '__main__':
    train_model()