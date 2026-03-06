
import json
import pandas as pd
from pathlib import Path

ATTRIBUTES = ['name', 'phone', 'website', 'address', 'category']
RESULTS_DIR = 'data/results'
OUTPUT_FILE = 'data/results/final_golden_dataset_2k_consolidated.json'

def consolidate():
    print("Consolidating results from all attributes...")
    
    # Dictionary to hold merged records: id -> { attribute: value }
    merged_data = {}
    
    for attr in ATTRIBUTES:
        if attr == 'name':
            candidates = [
                Path(RESULTS_DIR) / 'final_conflated_name_2k.json',
                Path(RESULTS_DIR) / 'final_conflated_names_2k.json',
            ]
        else:
            candidates = [Path(RESULTS_DIR) / f'final_conflated_{attr}_2k.json']

        filepath = next((p for p in candidates if p.exists()), None)
        
        if filepath is None:
            print(f"Warning: Missing results for {attr}")
            continue
            
        print(f"Loading {attr} results...")
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for record in data:
            rid = record['id']
            if rid not in merged_data:
                merged_data[rid] = {'id': rid, 'record_index': record['record_index']}
            
            # Store the chosen value and metadata
            conflated = record.get('conflated_value', record.get('conflated_name'))
            merged_data[rid][attr] = {
                'value': conflated,
                'source': record['selected_source'],
                'confidence': record['model_confidence']
            }

    # Convert to list
    final_records = list(merged_data.values())
    # Sort by record index
    final_records.sort(key=lambda x: x['record_index'])
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Consolidated {len(final_records)} records to {OUTPUT_FILE}")
    
    # Print stats
    print("\nGlobal Statistics (Source Selection):")
    for attr in ATTRIBUTES:
        curr = sum(1 for r in final_records if r.get(attr, {}).get('source') == 'current')
        base = sum(1 for r in final_records if r.get(attr, {}).get('source') == 'base')
        print(f"  {attr.upper()}: Current={curr}, Base={base}")

if __name__ == "__main__":
    consolidate()
