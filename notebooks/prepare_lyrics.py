"""
Quick script to convert lyrics folder to lyrics.csv
"""
import os
import glob
import pandas as pd

lyrics_dir = "data/lyrics"
output_csv = "data/lyrics.csv"

# Get all txt files
lyrics_files = sorted(glob.glob(os.path.join(lyrics_dir, "*.txt")))

data = []
for filepath in lyrics_files:
    filename = os.path.basename(filepath)
    # Extract id from filename (e.g., "001_Katy Perry - Roar_clip0.txt" -> "001_Katy Perry - Roar_clip0")
    file_id = os.path.splitext(filename)[0]
    
    # Read lyrics content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lyrics_text = f.read().strip()
    except:
        lyrics_text = ""
    
    data.append({
        'id': file_id,
        'filename': filename,
        'lyrics': lyrics_text
    })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Created {output_csv} with {len(df)} entries")
print(f"Sample IDs: {df['id'].head().tolist()}")
