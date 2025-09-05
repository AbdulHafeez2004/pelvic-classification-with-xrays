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
    
    print(f"Looking for images in: {os.path.abspath(base_path)}")
    
    for gender in ['male', 'female']:
        gender_path = os.path.join(base_path, gender)
        if not os.path.exists(gender_path):
            print(f"❌ Warning: {gender} folder does not exist at {gender_path}")
            continue
            
        print(f"📁 Scanning {gender} folder...")
        gender_files = os.listdir(gender_path)
        image_count = 0
        
        for img_file in gender_files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(gender_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(gender)
                    image_count += 1
                else:
                    print(f"   ❌ Could not read: {img_file}")
        
        print(f"   ✅ Loaded {image_count} images for {gender}")
    
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

def detect_pelvic_structures(img):
    """
    Try to detect pelvic bone structures in the image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for large, bone-like structures
    bone_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum size for bone structures
            bone_contours.append(contour)
    
    # Calculate bone structure coverage percentage
    total_area = gray.shape[0] * gray.shape[1]
    bone_area = sum(cv2.contourArea(c) for c in bone_contours)
    bone_coverage = bone_area / total_area
    
    return bone_coverage > 0.1  # At least 10% of image should show bone structures

def validate_pelvic_xray(image_path):
    """
    Strict validation specifically for pelvic X-ray images
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "File does not exist"
        
        # Check if we can read the image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read image file. Please upload a valid image."
        
        # Get image properties
        height, width = img.shape[:2]
        file_size = os.path.getsize(image_path) / 1024  # KB
        
        # Check 1: Minimum dimensions
        if height < 512 or width < 512:
            return False, f"Image dimensions ({width}x{height}) are too small for a pelvic X-ray. Minimum required: 512x512 pixels."
        
        # Check 2: Aspect ratio (pelvic X-rays are typically wider than tall)
        aspect_ratio = width / height
        if aspect_ratio < 0.8 or aspect_ratio > 2.5:  # More lenient range
            return False, f"Image aspect ratio ({aspect_ratio:.2f}) doesn't match typical pelvic X-rays (0.8-2.5)."
        
        # Check 3: File size (X-rays are usually larger files)
        if file_size < 30:  # Reduced from 50KB to 30KB
            return False, f"File size ({file_size:.1f}KB) is too small for a pelvic X-ray. Expected at least 30KB."
        
        # Check 4: Image is grayscale (X-rays are typically grayscale)
        if len(img.shape) == 3:
            # Calculate color variance
            color_variance = np.std(img, axis=2).mean()
            if color_variance > 25:  # Increased from 15 to 25
                return False, "Image appears to be a color photograph, not an X-ray."
        
        # Check 5: Histogram analysis - MADE MORE LENIENT
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Calculate intensity statistics
        intensity_5th = np.percentile(gray, 5)   # Very dark areas
        intensity_95th = np.percentile(gray, 95)  # Very bright areas
        intensity_range = intensity_95th - intensity_5th
        
        # X-rays typically have wide intensity range
        if intensity_range < 60:  # Reduced from 100 to 60
            return False, "Image lacks the wide intensity range typical of X-rays."
        
        # Check 6: Contrast (X-rays have high contrast)
        contrast = np.std(gray)
        if contrast < 35:  # Reduced from 40 to 35
            return False, "Image contrast is too low for an X-ray."
        
        # Check 7: Check for presence of both dark and bright areas
        dark_pixels = np.sum(gray < 50) / gray.size  # % of very dark pixels
        bright_pixels = np.sum(gray > 200) / gray.size  # % of very bright pixels
        
        if dark_pixels < 0.05 or bright_pixels < 0.02:  # Very lenient thresholds
            return False, "Image doesn't have the characteristic dark and bright areas of an X-ray."
        
        # If all checks pass
        return True, "Image appears to be a valid pelvic X-ray."
        
    except Exception as e:
        # Handle any unexpected errors gracefully
        return False, f"Error validating image: {str(e)}"