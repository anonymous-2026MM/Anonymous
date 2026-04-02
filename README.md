# VGD_CDR Model Usage Guide

## 1. Data Structure

### 1.1 Directory Layout
```
./data/
├── raw/                              # Raw data directory
│   ├── domain1_5.json.gz            # Rating data (Amazon format)
│   ├── domain2_5.json.gz            # Rating data for target domain
│   ├── domain1_image_extractor.jsonl  # Visual features (e.g., resnet, vit)
│   ├── domain2_image_extractor.jsonl # Visual features for target domain
│   └── domain1_text_extractor.jsonl   # Collaborative features 
├── mid/                              # Intermediate CSV files (auto-generated)
└── ready/                            # Ready-to-use datasets (auto-generated)
    └── _xx_xx/                        # Data split ratio (e.g., _8_2 for 80:20)
        └── tgt_domain2_src_domain1/   # Task-specific directory
            ├── train_src.csv
            ├── train_tgt.csv
            ├── train_meta.csv
            ├── test.csv
            └── id_map.json
```

### 1.2 File Formats

**Rating Data** (`domain1_5.json.gz`):
- Amazon Review and meta data: https://nijianmo.github.io/amazon/index.html
- Gzip-compressed JSON lines
- Fields per line: `reviewerID` (user ID), `asin` (item ID), `overall` (rating 1-5)

**Visual Features** (`domain1_image_xx.jsonl`):
- JSON lines format, one object per line
- Format: `{"item_id": [0.1, 0.2, ..., 0.9]}`
- Example: `{"B0002ZW0O6": [0.123, 0.456, ..., 0.789]}`

## 2. Running Pipeline

### Stage 1: Data Preprocessing (Required)

**Step 1 - Process Raw Data:**
```bash
python entry.py --process_data_mid 1 --root ./data/
```

**Step 2 - Generate Train/Test Splits:**
```bash
python entry.py --process_data_ready 1 --task x --root ./data/
```

*Note: Update `uid` and `iid` counts in `config.json` based on the output prompt.*

### Stage 2: Train Base Model  

**Option A - Base_Space (fixed-item-embedding):**
```bash
python entry.py --task x --epoch xx --lr xx --gpu x --exp_part Base_Space --save_path ./model_save/task_x.pth 
```

**Option B - None_CDR :**
```bash
python entry.py --task x --epoch xx --lr xx --gpu x --exp_part None_CDR --save_path ./model_save/task_x.pth 
```

### Stage 3: Train Domain-Invariant Extractor
```bash
python entry.py --task x --epoch xx --diff_lr xx --gpu x --exp_part Decouple --save_path ./model_save/task_x.pth 
```

*Note: Must use the same `--save_path` as Stage 2.*

### Stage 4: Train VGD_CDR Model
```bash
python entry.py --task x --epoch xx --diff_lr xx --gpu x --exp_part VGD_CDR --save_path ./model_save/task_x.pth --log_file ./logs/task_x.log 
```
