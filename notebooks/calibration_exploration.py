import os
import csv
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base_dir, 'data', 'features.csv')
    
    if not os.path.exists(features_path):
        print("features.csv not found. Please run src/setup_data.py first.")
        return

    features = []
    with open(features_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(row)
            
    if not features:
        print("features.csv is empty.")
        return

    healthy = [f for f in features if f['label'] == 'healthy']
    deficient = [f for f in features if f['label'] == 'deficient']
    possibly_deficient = [f for f in features if f['label'] == 'possibly_deficient']

    print(f"Loaded {len(healthy)} healthy, {len(deficient)} deficient, {len(possibly_deficient)} possibly_deficient samples.")

    # We will compute basic stats and print them to help set thresholds
    # Features to analyze: vein_density, yellow_pixel_ratio, excess_green_index, dgci, mean_saturation, vein_thickness_avg, interveinal_contrast
    keys_to_analyze = [
        'vein_density', 'yellow_pixel_ratio', 'excess_green_index', 'dgci', 
        'mean_saturation', 'vein_thickness_avg', 'interveinal_contrast'
    ]

    for key in keys_to_analyze:
        h_vals = [float(f[key]) for f in healthy if key in f]
        d_vals = [float(f[key]) for f in deficient if key in f]
        
        if not h_vals:
            continue
            
        h_mean = np.mean(h_vals)
        h_std = np.std(h_vals)
        d_mean = np.mean(d_vals) if d_vals else 0
        
        print(f"\n--- {key} ---")
        print(f"Healthy:   Mean = {h_mean:.4f}, Std = {h_std:.4f}")
        print(f"Deficient: Mean = {d_mean:.4f}")
        
        # Simple rule: if higher is healthier
        if h_mean > d_mean:
            thresh_low = h_mean - 1.5 * h_std
            thresh_def = h_mean - 2.5 * h_std
            print(f"Suggested Healthy Low Threshold: {thresh_low:.4f}")
            print(f"Suggested Deficient Threshold:   {thresh_def:.4f}")
        else:
            # Lower is healthier
            thresh_high = h_mean + 1.5 * h_std
            thresh_def = h_mean + 2.5 * h_std
            print(f"Suggested Healthy High Threshold: {thresh_high:.4f}")
            print(f"Suggested Deficient Threshold:    {thresh_def:.4f}")

if __name__ == '__main__':
    main()
