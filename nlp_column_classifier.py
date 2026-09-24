import os
import json
import re
import ast
import unicodedata
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import Counter

CAT_GENUINE = "GENUINE_LISTING_TEXT"
CAT_PLACEHOLDER = "PLACEHOLDER"
CAT_NEAR_EMPTY = "NEAR_EMPTY_OR_SHORT"
CAT_MISSING = "MISSING_OR_NULL"

PLACEHOLDER_REGEX = re.compile(
    r'^(n/?a|na|none|null|tbd|test|no description|dot|\.|\?|same as title|listing|apartment|house|place|see title)$',
    re.IGNORECASE
)

STANDARD_AMENITY_CATEGORIES = {
    "wifi": [r"\bwifi\b", r"\binternet\b"],
    "kitchen": [r"\bkitchen\b", r"\bkitchenette\b"],
    "air_conditioning": [r"\bair conditioning\b", r"\bac\b", r"\bcentral air\b"],
    "heating": [r"\bheating\b", r"\bheater\b"],
    "free_parking": [r"\bfree parking\b", r"\bdriveway\b", r"\bcarport\b"],
    "paid_parking": [r"\bpaid parking\b", r"\bparking on premises\b"],
    "washer": [r"\bwasher\b", r"\blaundry\b"],
    "dryer": [r"\bdryer\b"],
    "tv": [r"\btv\b", r"\bhdtv\b", r"\btelevision\b"],
    "dedicated_workspace": [r"\bworkspace\b", r"\bdesk\b"],
    "smoke_alarm": [r"\bsmoke alarm\b", r"\bsmoke detector\b"],
    "carbon_monoxide_alarm": [r"\bcarbon monoxide\b"],
    "dishwasher": [r"\bdishwasher\b"],
    "microwave": [r"\bmicrowave\b"],
    "refrigerator": [r"\brefrigerator\b", r"\bfridge\b"],
    "coffee_maker": [r"\bcoffee\b", r"\bespresso\b"],
    "patio_or_balcony": [r"\bpatio\b", r"\bbalcony\b"],
    "bbq_grill": [r"\bbbq\b", r"\bbarbecue\b", r"\bgrill\b"],
    "pool_or_hot_tub": [r"\bpool\b", r"\bhot tub\b"]
}

def classify_text_field(text):
    if pd.isna(text) or text is None:
        return CAT_MISSING, 0.0, {
            "word_count": 0,
            "char_count": 0,
            "reason": "Missing or NaN text field"
        }
        
    s = str(text).strip()
    if not s:
        return CAT_MISSING, 0.0, {
            "word_count": 0,
            "char_count": 0,
            "reason": "Empty text string"
        }
        
    words = s.split()
    word_count = len(words)
    char_count = len(s)
    clean_lower = re.sub(r'[^\w\s]', '', s.lower()).strip()
    
    if PLACEHOLDER_REGEX.match(clean_lower) or PLACEHOLDER_REGEX.match(s.lower()):
        return CAT_PLACEHOLDER, 0.05, {
            "word_count": word_count,
            "char_count": char_count,
            "reason": f"Matches placeholder pattern '{clean_lower}'"
        }
        
    if word_count == 1:
        if len(clean_lower) < 15 or clean_lower in ['apartment', 'house', 'studio', 'room', 'listing']:
            return CAT_PLACEHOLDER, 0.10, {
                "word_count": 1,
                "char_count": char_count,
                "reason": f"Single-word non-descriptive listing text ('{s}')"
            }
            
    if word_count <= 5 or char_count < 25:
        return CAT_NEAR_EMPTY, 0.35, {
            "word_count": word_count,
            "char_count": char_count,
            "reason": f"Near-empty description (word_count={word_count}, char_count={char_count})"
        }
        
    unique_words = len(set(w.lower() for w in words))
    diversity = round(unique_words / word_count, 3) if word_count > 0 else 0.0
    confidence = min(1.0, round(0.5 + (word_count / 100.0) * 0.5, 3))
    
    return CAT_GENUINE, confidence, {
        "word_count": word_count,
        "char_count": char_count,
        "lexical_diversity": diversity,
        "reason": "Substantive listing text"
    }

def clean_amenity_string(item):
    if not isinstance(item, str):
        item = str(item)
        
    s = item.replace('\ufffd', "'")
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u202f', ' ').replace('\xa0', ' ')
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = s.strip('"\' ')
    return s

def parse_and_standardize_amenities(raw_amenities):
    if pd.isna(raw_amenities) or raw_amenities is None:
        return [], "MISSING", ["Null or NaN amenities field"]
        
    s = str(raw_amenities).strip()
    if not s or s in ['[]', '{}']:
        return [], "EMPTY", ["Empty amenities array"]
        
    items = []
    format_type = "JSON"
    issues = []
    
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [str(parsed)]
    except Exception:
        try:
            parsed = ast.literal_eval(s)
            format_type = "PYTHON_LITERAL"
            if isinstance(parsed, (list, set, tuple)):
                items = list(parsed)
            else:
                items = [str(parsed)]
        except Exception:
            format_type = "PLAIN_TEXT_COMMA"
            items = [x.strip() for x in s.split(',') if x.strip()]
            issues.append("Fallback comma parsing required")
            
    cleaned_items = []
    for item in items:
        raw_str = str(item)
        if any(ord(c) > 127 for c in raw_str) or '\\u' in raw_str or '\ufffd' in raw_str:
            if "contains_encoding_artifacts" not in issues:
                issues.append("contains_encoding_artifacts")
        cleaned = clean_amenity_string(item)
        if cleaned:
            cleaned_items.append(cleaned)
            
    if len(cleaned_items) != len(set(cleaned_items)):
        issues.append("contains_duplicate_items")
        seen = set()
        deduped = []
        for x in cleaned_items:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        cleaned_items = deduped
        
    return cleaned_items, format_type, issues

def run_nlp_classification(input_path="cleaned_data.csv", output_path="nlp_classification_report.json"):
    print("==================================================")
    print("RUNNING NLP COLUMN CLASSIFIER & AMENITIES PARSER")
    print("==================================================")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found at: {input_path}")
        
    df = pd.read_csv(input_path, low_memory=False)
    total_rows = len(df)
    print(f"Loaded dataset from {input_path} with {total_rows} rows.")
    
    text_cols = ['description', 'neighborhood_overview']
    text_summary = {}
    flagged_text_issues = []
    
    for col in text_cols:
        cat_counts = Counter()
        confidences = []
        word_counts = []
        char_counts = []
        
        if col not in df.columns:
            text_summary[col] = {
                "total_rows": total_rows,
                "total_non_null": 0,
                "total_null": total_rows,
                "counts_by_category": {CAT_MISSING: total_rows, CAT_GENUINE: 0, CAT_PLACEHOLDER: 0, CAT_NEAR_EMPTY: 0}
            }
            continue
            
        for idx in range(total_rows):
            val = df.loc[idx, col]
            cat, conf, meta = classify_text_field(val)
            cat_counts[cat] += 1
            confidences.append(conf)
            word_counts.append(meta["word_count"])
            char_counts.append(meta["char_count"])
            
            if cat in [CAT_PLACEHOLDER, CAT_NEAR_EMPTY, CAT_MISSING]:
                listing_id = int(df.loc[idx, 'id']) if 'id' in df.columns and pd.notna(df.loc[idx, 'id']) else idx
                flagged_text_issues.append({
                    "row_index": idx,
                    "listing_id": listing_id,
                    "column": col,
                    "text_sample": str(val)[:100] if pd.notna(val) else None,
                    "classification_category": cat,
                    "confidence_score": conf,
                    "meta": meta
                })
                
        non_null_count = int(df[col].notna().sum())
        text_summary[col] = {
            "total_rows": total_rows,
            "total_non_null": non_null_count,
            "total_null": int(df[col].isna().sum()),
            "counts_by_category": dict(cat_counts),
            "percentages_by_category": {
                k: round(float(v / total_rows * 100), 2) for k, v in cat_counts.items()
            },
            "mean_word_count": round(float(np.mean(word_counts)), 2),
            "mean_char_count": round(float(np.mean(char_counts)), 2),
            "mean_confidence_score": round(float(np.mean(confidences)), 3)
        }
        
    amenities_summary = {
        "total_processed": total_rows,
        "valid_json_count": 0,
        "corrupted_encoding_flagged_count": 0,
        "duplicate_items_flagged_count": 0,
        "empty_amenities_count": 0
    }
    
    all_cleaned_amenities = []
    amenity_counts_per_row = []
    amenities_freq = Counter()
    category_matches = Counter()
    flagged_amenities_issues = []
    
    amenities_col = 'amenities' if 'amenities' in df.columns else None
    
    if amenities_col:
        for idx in range(total_rows):
            raw_val = df.loc[idx, amenities_col]
            cleaned_items, format_type, issues = parse_and_standardize_amenities(raw_val)
            
            if format_type == "JSON":
                amenities_summary["valid_json_count"] += 1
            if "contains_encoding_artifacts" in issues:
                amenities_summary["corrupted_encoding_flagged_count"] += 1
            if "contains_duplicate_items" in issues:
                amenities_summary["duplicate_items_flagged_count"] += 1
            if not cleaned_items:
                amenities_summary["empty_amenities_count"] += 1
                
            amenity_counts_per_row.append(len(cleaned_items))
            for item in cleaned_items:
                amenities_freq[item] += 1
                all_cleaned_amenities.append(item)
                
                item_lower = item.lower()
                for cat_key, patterns in STANDARD_AMENITY_CATEGORIES.items():
                    if any(re.search(pat, item_lower) for pat in patterns):
                        category_matches[cat_key] += 1
                        
            if issues:
                listing_id = int(df.loc[idx, 'id']) if 'id' in df.columns and pd.notna(df.loc[idx, 'id']) else idx
                flagged_amenities_issues.append({
                    "row_index": idx,
                    "listing_id": listing_id,
                    "format_type": format_type,
                    "amenity_count": len(cleaned_items),
                    "issues": issues
                })
                
    top_20_amenities = dict(amenities_freq.most_common(20))
    
    amenities_parsing_report = {
        "summary": amenities_summary,
        "metrics": {
            "avg_amenities_per_listing": round(float(np.mean(amenity_counts_per_row)), 2) if amenity_counts_per_row else 0.0,
            "min_amenities_per_listing": int(np.min(amenity_counts_per_row)) if amenity_counts_per_row else 0,
            "max_amenities_per_listing": int(np.max(amenity_counts_per_row)) if amenity_counts_per_row else 0,
            "total_unique_standardized_amenities": len(amenities_freq)
        },
        "top_20_amenities": top_20_amenities,
        "standard_category_counts": dict(category_matches)
    }
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": input_path,
        "total_rows_evaluated": total_rows,
        "text_classification_summary": text_summary,
        "amenities_parsing_report": amenities_parsing_report,
        "flagged_text_issues_count": len(flagged_text_issues),
        "flagged_amenities_issues_count": len(flagged_amenities_issues),
        "flagged_text_issues": flagged_text_issues,
        "flagged_amenities_issues": flagged_amenities_issues
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"PASS: Text columns classified across {total_rows} rows.")
    print(f"      Description categories: {text_summary['description']['counts_by_category']}")
    print(f"PASS: Amenities parsed and standardized across {total_rows} rows.")
    print(f"      Corrupted encoding flagged: {amenities_summary['corrupted_encoding_flagged_count']} listings.")
    print(f"      Average amenities per listing: {amenities_parsing_report['metrics']['avg_amenities_per_listing']}")
    print(f"Report written to {output_path}.")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Column Classifier & Amenities Parser")
    parser.add_argument("--input", default="cleaned_data.csv", help="Input dataset CSV")
    parser.add_argument("--output", default="nlp_classification_report.json", help="Output JSON report path")
    args = parser.parse_args()
    run_nlp_classification(args.input, args.output)
