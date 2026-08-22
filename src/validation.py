import os
import csv
from pipeline import run_pipeline_single_image
from decision_engine import VERDICT_HEALTHY, VERDICT_POSSIBLY_DEFICIENT, VERDICT_DEFICIENT

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    val_dir = os.path.join(base_dir, 'data', 'validation')
    labels_path = os.path.join(base_dir, 'data', 'labels.csv')
    
    if not os.path.exists(labels_path):
        print("labels.csv not found.")
        return

    # Load ground truth labels
    labels = {}
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['image_id']] = row['label']

    if not os.path.exists(val_dir) or not os.listdir(val_dir):
        print("Validation directory is empty or missing.")
        return

    print("--- VALIDATION RESULTS ---")
    
    y_true = []
    y_pred = []
    
    # Map raw labels to verdict strings
    label_map = {
        'healthy': VERDICT_HEALTHY,
        'possibly_deficient': VERDICT_POSSIBLY_DEFICIENT,
        'deficient': VERDICT_DEFICIENT
    }

    correct = 0
    total = 0

    confusion = {
        VERDICT_HEALTHY: {VERDICT_HEALTHY: 0, VERDICT_POSSIBLY_DEFICIENT: 0, VERDICT_DEFICIENT: 0},
        VERDICT_POSSIBLY_DEFICIENT: {VERDICT_HEALTHY: 0, VERDICT_POSSIBLY_DEFICIENT: 0, VERDICT_DEFICIENT: 0},
        VERDICT_DEFICIENT: {VERDICT_HEALTHY: 0, VERDICT_POSSIBLY_DEFICIENT: 0, VERDICT_DEFICIENT: 0}
    }

    for img_name in sorted(os.listdir(val_dir)):
        if not img_name.endswith('.jpeg'):
            continue
            
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(val_dir, img_name)
        
        true_label = labels.get(img_id)
        if not true_label:
            continue
            
        true_verdict = label_map.get(true_label, VERDICT_HEALTHY)
        
        try:
            res = run_pipeline_single_image(img_path, img_id, save_debug=False)
            pred_verdict = res['verdict_result']['verdict']
            
            y_true.append(true_verdict)
            y_pred.append(pred_verdict)
            
            confusion[true_verdict][pred_verdict] += 1
            
            total += 1
            if true_verdict == pred_verdict:
                correct += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("\n--- Confusion Matrix ---")
    print("True \\ Pred | Healthy | Possibly Def | Deficient")
    for t_verdict in [VERDICT_HEALTHY, VERDICT_POSSIBLY_DEFICIENT, VERDICT_DEFICIENT]:
        h_pred = confusion[t_verdict][VERDICT_HEALTHY]
        p_pred = confusion[t_verdict][VERDICT_POSSIBLY_DEFICIENT]
        d_pred = confusion[t_verdict][VERDICT_DEFICIENT]
        
        # Format true label padded to 10 chars
        t_str = t_verdict[:10].ljust(10)
        print(f"{t_str}  | {h_pred:7} | {p_pred:12} | {d_pred:9}")
        
    print("\n--- Metrics ---")
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
if __name__ == '__main__':
    main()
