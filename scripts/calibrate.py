import os
import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import resize_to_working_resolution, normalize_brightness, denoise
from src.segmentation import segment_leaf
from src.vein_extraction import extract_veins
from src.feature_extraction import extract_all_features

def extract_features_for_dataset(data_dir: str, labels_file: str, output_csv: str) -> pd.DataFrame:
    """Extract features from all images in reference dataset and save to CSV."""
    if not os.path.exists(labels_file):
        raise FileNotFoundError("No labels.csv found.")
        
    labels_df = pd.read_csv(labels_file)
    features_list = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        image_id = filename.split('.')[0]
        img_path = os.path.join(data_dir, filename)
        
        # Get label
        label_row = labels_df[labels_df['image_id'] == int(image_id)]
        if label_row.empty:
            continue
        label = label_row.iloc[0]['label']
        
        img_raw = cv2.imread(img_path)
        img_resized = resize_to_working_resolution(img_raw)
        img_preprocessed = normalize_brightness(denoise(img_resized))
        
        seg_result = segment_leaf(img_preprocessed)
        mask = seg_result['mask']
        leaf_area = cv2.countNonZero(mask)
        
        if leaf_area == 0:
            continue
            
        vein_result = extract_veins(img_preprocessed, mask)
        feats = extract_all_features(img_preprocessed, mask, vein_result, leaf_area)
        
        feats['image_id'] = image_id
        feats['label'] = label
        features_list.append(feats)
        
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    return df


def plot_distributions(df: pd.DataFrame, output_dir: str):
    """Plot feature distributions for healthy vs deficient."""
    os.makedirs(output_dir, exist_ok=True)
    
    features_to_plot = [
        'vein_density', 'vein_thickness_avg', 'yellow_pixel_ratio', 
        'excess_green_index', 'dgci', 'interveinal_contrast', 'color_spatial_variance'
    ]
    
    for feature in features_to_plot:
        if feature not in df.columns:
            continue
            
        plt.figure(figsize=(8, 5))
        healthy = df[df['label'] == 'healthy'][feature]
        deficient = df[df['label'] != 'healthy'][feature]
        
        # Use scatter plot since N is very small (N=5)
        plt.scatter(np.zeros_like(healthy), healthy, color='green', label='Healthy', alpha=0.7)
        plt.scatter(np.ones_like(deficient), deficient, color='red', label='Deficient', alpha=0.7)
        
        plt.title(f"{feature} Distribution")
        plt.xticks([0, 1], ['Healthy', 'Deficient'])
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(output_dir, f"{feature}_distribution.png"))
        plt.close()


def update_thresholds_file(thresholds_path: str, new_thresholds: dict):
    """Dynamically rewrite config/thresholds.py with calculated values."""
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out_lines = []
    for line in lines:
        updated = False
        for key, val in new_thresholds.items():
            if line.startswith(f"{key} = "):
                if isinstance(val, float):
                    out_lines.append(f"{key} = {val:.6f}\n")
                else:
                    out_lines.append(f"{key} = {val}\n")
                updated = True
                break
        if not updated:
            out_lines.append(line)
            
    with open(thresholds_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)


def main():
    data_dir = os.path.join("data", "reference")
    labels_file = os.path.join("data", "labels.csv")
    features_file = os.path.join("data", "features.csv")
    plots_dir = os.path.join("output", "calibration_plots")
    thresholds_path = os.path.join("config", "thresholds.py")
    
    print("Extracting features (with glare masking)...")
    df = extract_features_for_dataset(data_dir, labels_file, features_file)
    print(f"Extracted features for {len(df)} images.")
    
    print("Plotting distributions...")
    plot_distributions(df, plots_dir)
    
    healthy_df = df[df['label'] == 'healthy']
    deficient_df = df[df['label'] != 'healthy']
    
    if healthy_df.empty or deficient_df.empty:
        print("Error: Need both healthy and deficient samples to calculate thresholds.")
        return
        
    print("\n--- Calibration Results ---")
    new_thresholds = {}
    
    def get_std(series):
        std = series.std()
        return 0.0 if pd.isna(std) else std

    # 1. Yellow Ratio (Fail if >)
    h_mean = healthy_df['yellow_pixel_ratio'].mean()
    d_mean = deficient_df['yellow_pixel_ratio'].mean()
    h_std = get_std(healthy_df['yellow_pixel_ratio'])
    thresh_yr = h_mean + 1.5 * h_std if h_std > 0 else max(h_mean * 1.5, (h_mean + d_mean) / 2)
    if thresh_yr < 0.05: thresh_yr = 0.05
    new_thresholds['YELLOW_RATIO_DEFICIENT'] = float(thresh_yr)
    print(f"YELLOW_RATIO_DEFICIENT: {thresh_yr:.4f} (Healthy Mean: {h_mean:.4f}, Std: {h_std:.4f})")
    
    # 2. ExG (Fail if <)
    h_mean = healthy_df['excess_green_index'].mean()
    d_mean = deficient_df['excess_green_index'].mean()
    h_std = get_std(healthy_df['excess_green_index'])
    thresh_exg = h_mean - 1.5 * h_std if h_std > 0 else min(h_mean * 0.9, (h_mean + d_mean) / 2)
    new_thresholds['EXG_HEALTHY_LOW'] = float(thresh_exg)
    print(f"EXG_HEALTHY_LOW: {thresh_exg:.4f} (Healthy Mean: {h_mean:.4f}, Std: {h_std:.4f})")
    
    # 3. DGCI (Fail if <)
    h_mean = healthy_df['dgci'].mean()
    d_mean = deficient_df['dgci'].mean()
    h_std = get_std(healthy_df['dgci'])
    thresh_dgci = h_mean - 1.5 * h_std if h_std > 0 else min(h_mean * 0.9, (h_mean + d_mean) / 2)
    new_thresholds['DGCI_HEALTHY_LOW'] = float(thresh_dgci)
    print(f"DGCI_HEALTHY_LOW: {thresh_dgci:.4f} (Healthy Mean: {h_mean:.4f}, Std: {h_std:.4f})")
    
    # 4. Interveinal Contrast (Fail if >)
    h_mean = healthy_df['interveinal_contrast'].mean()
    d_mean = deficient_df['interveinal_contrast'].mean()
    h_std = get_std(healthy_df['interveinal_contrast'])
    thresh_iv = h_mean + 1.5 * h_std if h_std > 0 else max(h_mean * 1.2, (h_mean + d_mean) / 2)
    new_thresholds['INTERVEINAL_CONTRAST_THRESHOLD'] = float(thresh_iv)
    print(f"INTERVEINAL_CONTRAST_THRESHOLD: {thresh_iv:.4f} (Healthy Mean: {h_mean:.4f}, Std: {h_std:.4f})")
    
    # 5. Color Spatial Variance (Fail if >)
    h_mean = healthy_df['color_spatial_variance'].mean()
    d_mean = deficient_df['color_spatial_variance'].mean()
    h_std = get_std(healthy_df['color_spatial_variance'])
    thresh_var = h_mean + 1.5 * h_std if h_std > 0 else max(h_mean * 1.2, (h_mean + d_mean) / 2)
    new_thresholds['COLOR_SPATIAL_VARIANCE_MAX'] = float(thresh_var)
    print(f"COLOR_SPATIAL_VARIANCE_MAX: {thresh_var:.4f} (Healthy Mean: {h_mean:.4f}, Std: {h_std:.4f})")
    
    # 6. Vein Density (Fail if <)
    h_mean = healthy_df['vein_density'].mean()
    d_mean = deficient_df['vein_density'].mean()
    h_std = get_std(healthy_df['vein_density'])
    thresh_vd = h_mean - 1.5 * h_std if h_std > 0 else min(h_mean * 0.9, (h_mean + d_mean) / 2)
    new_thresholds['VEIN_DENSITY_DEFICIENT'] = float(thresh_vd)
    print(f"VEIN_DENSITY_DEFICIENT: {thresh_vd:.6f} (Healthy Mean: {h_mean:.4f}, Std: {h_std:.4f})")
    
    # 7. Vein Thickness (Fail if >)
    h_mean = healthy_df['vein_thickness_avg'].mean()
    d_mean = deficient_df['vein_thickness_avg'].mean()
    h_std = get_std(healthy_df['vein_thickness_avg'])
    thresh_vt = h_mean + 1.5 * h_std if h_std > 0 else max(h_mean * 1.1, (h_mean + d_mean) / 2)
    new_thresholds['VEIN_THICKNESS_DEFICIENT_HIGH'] = float(thresh_vt)
    print(f"VEIN_THICKNESS_DEFICIENT_HIGH: {thresh_vt:.4f} (Healthy Mean: {h_mean:.4f}, Std: {h_std:.4f})")
    
    print("\nUpdating config/thresholds.py with new calibrated values...")
    update_thresholds_file(thresholds_path, new_thresholds)
    print("Done.")

if __name__ == '__main__':
    main()
