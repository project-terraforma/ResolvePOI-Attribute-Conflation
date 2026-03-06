
import json
import pandas as pd
import argparse
try:
    from scripts.extract_features import extract_features_batch
except ModuleNotFoundError:
    from extract_features import extract_features_batch

INPUT_FILE = 'data/synthetic_golden_dataset_2k.json'

def process_synthetic():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', default='name', choices=['name', 'phone', 'website', 'address', 'category'])
    parser.add_argument(
        '--include-metadata',
        choices=['true', 'false'],
        default='false',
        help='Whether to include metadata features (confidence, sources) in training features.',
    )
    parser.add_argument(
        '--use-record-confidence',
        choices=['true', 'false'],
        default='true',
        help='Whether to use synthetic record confidence values, else use fixed 1.0/0.5.',
    )
    args = parser.parse_args()

    include_metadata = args.include_metadata == 'true'
    use_record_confidence = args.use_record_confidence == 'true'
    
    output_features = f'data/processed/features_{args.attribute}_synthetic.parquet'
    
    print(f"Loading synthetic dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        records = json.load(f)
        
    # Convert to DataFrame format expected by feature extractor
    df_rows = []
    labels = {}
    
    for r in records:
        if use_record_confidence:
            confidence = float(r['data']['current'].get('confidence', 1.0))
            base_confidence = float(r['data']['base'].get('confidence', 0.5))
        else:
            confidence = 1.0
            base_confidence = 0.5

        row = {
            'id': r['id'],
            'names': r['data']['current']['names'],
            'base_names': r['data']['base']['names'],
            'phones': r['data']['current']['phones'],
            'base_phones': r['data']['base']['phones'],
            'websites': r['data']['current']['websites'],
            'base_websites': r['data']['base']['websites'],
            'addresses': r['data']['current']['addresses'],
            'base_addresses': r['data']['base']['addresses'],
            'categories': r['data']['current']['categories'],
            'base_categories': r['data']['base']['categories'],
            'confidence': confidence,
            'base_confidence': base_confidence,
            'sources': '',
            'base_sources': ''
        }
        df_rows.append(row)
        labels[r['id']] = r['label']
        
    df = pd.DataFrame(df_rows)
    print(f"Converted {len(df)} records to DataFrame.")
    
    # Extract Features
    print(f"Extracting features for '{args.attribute}'...")
    features_df = extract_features_batch(
        df,
        attribute=args.attribute,
        include_metadata=include_metadata,
    )
    
    # Add Labels
    print("Applying labels...")
    features_df['label'] = features_df['id'].map(labels)
    
    # Save
    features_df.to_parquet(output_features, index=False)
    print(f"✅ Saved features with labels to {output_features}")

if __name__ == "__main__":
    process_synthetic()
