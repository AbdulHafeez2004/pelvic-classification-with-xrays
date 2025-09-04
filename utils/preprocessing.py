import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import os

def load_images_from_directory(base_path, image_size=(224, 224)):
    """
    Load images from directory structure: base_path/gender/*.jpg
    """
    images = []
    labels = []
    
    for gender in ['male', 'female']:
        gender_path = os.path.join(base_path, gender)
        if os.path.exists(gender_path):
            for img_file in os.listdir(gender_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(gender_path, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        img = img.astype('float32') / 255.0
                        images.append(img)
                        labels.append(gender)
    
    return np.array(images), np.array(labels)

def extract_features(images):
    """
    Extract HOG and LBP features from images
    """
    hog_features = []
    lbp_features = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # HOG features
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        hog_features.append(hog_feat)
        
        # LBP features
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_features.append(hist)
    
    hog_features = np.array(hog_features)
    lbp_features = np.array(lbp_features)
    combined_features = np.hstack([hog_features, lbp_features])
    
    return combined_features

def preprocess_single_image(image_path, image_size=(224, 224)):
    """
    Preprocess a single image for prediction
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, image_size)
    img = img.astype('float32') / 255.0
    return img

def extract_single_image_features(img):
    """
    Extract features from a single image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG features
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    
    # LBP features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    
    combined_features = np.hstack([hog_feat, hist])
    return combined_features.reshape(1, -1)