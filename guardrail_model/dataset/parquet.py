import os
import pandas as pd

RAW_DIR = "data/raw"

def convert_all():
    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            out_path = os.path.splitext(file_path)[0] + ".parquet"
            
            if file.endswith(".csv"):
                print(f"Converting CSV: {file}")
                df = pd.read_csv(file_path, low_memory=False)
            elif file.endswith(".jsonl") or file.endswith(".json"):
                print(f"Converting JSON: {file}")
                is_lines = file.endswith(".jsonl")
                df = pd.read_json(file_path, lines=is_lines)
            else:
                continue
                
            # Convert any columns containing lists or dicts to strings to satisfy PyArrow
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df[col] = df[col].astype(str)
                    
            df.to_parquet(out_path, index=False)
            print(f"Successfully saved: {out_path}")

if __name__ == "__main__":
    convert_all()
    print("Conversion complete!")