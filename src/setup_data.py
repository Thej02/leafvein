import os
import csv
import shutil
import glob
from pipeline import run_pipeline_single_image

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    ref_dir = os.path.join(data_dir, 'reference')
    val_dir = os.path.join(data_dir, 'validation')
    labels_path = os.path.join(data_dir, 'labels.csv')
    features_path = os.path.join(data_dir, 'features.csv')

    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 1. Create labels.csv
    labels = [
        ['image_id', 'capture_date', 'label', 'ground_truth_basis', 'notes'],
        ['1', '2026-08-20', 'healthy', 'Visual grading against chart', 'Sample image'],
        ['2', '2026-08-20', 'healthy', 'Visual grading against chart', 'Sample image'],
        ['3', '2026-08-20', 'healthy', 'Visual grading against chart', 'Sample image'],
        ['4', '2026-08-20', 'possibly_deficient', 'Visual grading against chart', 'Sample image'],
        ['5', '2026-08-20', 'deficient', 'Visual grading against chart', 'Sample image'],
        ['6', '2026-08-20', 'healthy', 'Visual grading against chart', 'Sample image'],
        ['7', '2026-08-20', 'possibly_deficient', 'Visual grading against chart', 'Sample image'],
        ['8', '2026-08-20', 'deficient', 'Visual grading against chart', 'Sample image'],
    ]
    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(labels)

    # 2. Split into reference and validation
    for i in range(1, 9):
        src = os.path.join(raw_dir, f'{i}.jpeg')
        if not os.path.exists(src):
            continue
        if i <= 5:
            dst = os.path.join(ref_dir, f'{i}.jpeg')
        else:
            dst = os.path.join(val_dir, f'{i}.jpeg')
        shutil.copy2(src, dst)

    # 3. Generate features.csv for reference dataset
    feature_rows = []
    headers = [
        'image_id', 'label', 'vein_density', 'vein_thickness_avg', 'branch_point_count',
        'vein_pixel_count', 'leaf_area_pixels', 'mean_hue', 'mean_saturation',
        'yellow_pixel_ratio', 'excess_green_index', 'dgci', 'interveinal_contrast'
    ]
    
    label_dict = {row[0]: row[2] for row in labels[1:]}

    # Generate features for all images in raw for calibration/validation
    for img_path in glob.glob(os.path.join(raw_dir, '*.jpeg')):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        try:
            res = run_pipeline_single_image(img_path, img_id, save_debug=False)
            features = res['features']
            row = [
                img_id,
                label_dict.get(img_id, 'unknown'),
                features['vein_density'],
                features['vein_thickness_avg'],
                features['branch_point_count'],
                features['vein_pixel_count'],
                features['leaf_area_pixels'],
                features['mean_hue'],
                features['mean_saturation'],
                features['yellow_pixel_ratio'],
                features['excess_green_index'],
                features['dgci'],
                features['interveinal_contrast']
            ]
            feature_rows.append(row)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    with open(features_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(feature_rows)

    print("Data setup complete.")

if __name__ == '__main__':
    main()
