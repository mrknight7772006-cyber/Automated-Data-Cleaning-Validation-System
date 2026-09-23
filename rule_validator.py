import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

class RuleValidator:
    """
    Rule-Based Validation Engine for Airbnb dataset (Sprint 7 / Module 3).
    Evaluates:
    1. Numeric Range Rules (price > 0, minimum_nights in [1,365], availability_365 in [0,365])
    2. Categorical Consistency Rules (room_type in valid set, neighbourhood_group_cleansed in valid city wards/boroughs)
    3. Coordinate Sanity Rule (latitude/longitude within city real bounding box)
    4. Cross-Field Consistency Rule (maximum_nights >= minimum_nights)
    """

    def __init__(self, city_name: str = "Albany, NY"):
        self.city_name = city_name
        self.valid_room_types = {"Entire home/apt", "Private room", "Hotel room", "Shared room"}
        self.valid_city_boroughs = {
            "FIRST WARD", "SECOND WARD", "THIRD WARD", "FOURTH WARD", "FIFTH WARD",
            "SIXTH WARD", "SEVENTH WARD", "EIGHTH WARD", "NINTH WARD", "TENTH WARD",
            "ELEVENTH WARD", "TWELFTH WARD", "THIRTEENTH WARD", "FOURTEENTH WARD", "FIFTEENTH WARD"
        }
        # City real bounding box for Albany, NY
        self.city_bbox = {
            "min_lat": 42.64,
            "max_lat": 42.71,
            "min_lon": -73.85,
            "max_lon": -73.74
        }

    def validate_dataframe(self, df: pd.DataFrame, dataset_path: str = "cleaned_data.csv") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        total_rows = len(df)
        violations_log: List[Dict[str, Any]] = []

        per_rule_counts = {
            "price_greater_than_zero": 0,
            "minimum_nights_in_range": 0,
            "availability_365_in_range": 0,
            "room_type_valid": 0,
            "neighbourhood_group_valid": 0,
            "coordinate_bounding_box": 0,
            "min_nights_le_max_nights": 0
        }

        for idx, row in df.iterrows():
            row_id = row.get("id", idx)

            # 1. Price > 0
            price_val = pd.to_numeric(str(row.get("price", "")).replace("$", "").replace(",", ""), errors="coerce")
            if pd.isna(price_val) or price_val <= 0:
                per_rule_counts["price_greater_than_zero"] += 1
                violations_log.append({
                    "rule_id": "R001",
                    "rule_name": "price_greater_than_zero",
                    "row_index": int(idx),
                    "listing_id": row_id,
                    "column": "price",
                    "value": str(row.get("price")),
                    "details": f"Price value '{row.get('price')}' is non-positive or invalid (must be > 0)."
                })

            # 2. minimum_nights in [1, 365]
            min_n = pd.to_numeric(row.get("minimum_nights"), errors="coerce")
            if pd.isna(min_n) or min_n < 1 or min_n > 365:
                per_rule_counts["minimum_nights_in_range"] += 1
                violations_log.append({
                    "rule_id": "R002",
                    "rule_name": "minimum_nights_in_range",
                    "row_index": int(idx),
                    "listing_id": row_id,
                    "column": "minimum_nights",
                    "value": str(row.get("minimum_nights")),
                    "details": f"minimum_nights '{row.get('minimum_nights')}' is outside valid range [1, 365]."
                })

            # 3. availability_365 in [0, 365]
            avail = pd.to_numeric(row.get("availability_365"), errors="coerce")
            if pd.isna(avail) or avail < 0 or avail > 365:
                per_rule_counts["availability_365_in_range"] += 1
                violations_log.append({
                    "rule_id": "R003",
                    "rule_name": "availability_365_in_range",
                    "row_index": int(idx),
                    "listing_id": row_id,
                    "column": "availability_365",
                    "value": str(row.get("availability_365")),
                    "details": f"availability_365 '{row.get('availability_365')}' is outside valid range [0, 365]."
                })

            # 4. room_type valid set
            room_type = str(row.get("room_type", "")).strip()
            if room_type not in self.valid_room_types:
                per_rule_counts["room_type_valid"] += 1
                violations_log.append({
                    "rule_id": "R004",
                    "rule_name": "room_type_valid",
                    "row_index": int(idx),
                    "listing_id": row_id,
                    "column": "room_type",
                    "value": room_type,
                    "details": f"room_type '{room_type}' is not in allowed set {sorted(list(self.valid_room_types))}."
                })

            # 5. neighbourhood_group_cleansed valid borough/ward list
            ngc = str(row.get("neighbourhood_group_cleansed", "")).strip().upper()
            if pd.isna(row.get("neighbourhood_group_cleansed")) or ngc not in self.valid_city_boroughs:
                per_rule_counts["neighbourhood_group_valid"] += 1
                violations_log.append({
                    "rule_id": "R005",
                    "rule_name": "neighbourhood_group_valid",
                    "row_index": int(idx),
                    "listing_id": row_id,
                    "column": "neighbourhood_group_cleansed",
                    "value": str(row.get("neighbourhood_group_cleansed")),
                    "details": f"neighbourhood_group_cleansed '{row.get('neighbourhood_group_cleansed')}' is not in valid borough/ward list for {self.city_name}."
                })

            # 6. Coordinate Sanity (latitude/longitude in city real bounding box)
            lat = pd.to_numeric(row.get("latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon) or lat < self.city_bbox["min_lat"] or lat > self.city_bbox["max_lat"] or lon < self.city_bbox["min_lon"] or lon > self.city_bbox["max_lon"]:
                per_rule_counts["coordinate_bounding_box"] += 1
                violations_log.append({
                    "rule_id": "R006",
                    "rule_name": "coordinate_bounding_box",
                    "row_index": int(idx),
                    "listing_id": row_id,
                    "column": "latitude,longitude",
                    "value": f"({row.get('latitude')}, {row.get('longitude')})",
                    "details": f"Coordinates ({lat}, {lon}) fall outside real city bounding box for {self.city_name} [lat: {self.city_bbox['min_lat']}..{self.city_bbox['max_lat']}, lon: {self.city_bbox['min_lon']}..{self.city_bbox['max_lon']}]."
                })

            # 7. Cross-field: maximum_nights >= minimum_nights
            max_n = pd.to_numeric(row.get("maximum_nights"), errors="coerce")
            if pd.isna(min_n) or pd.isna(max_n) or min_n > max_n:
                per_rule_counts["min_nights_le_max_nights"] += 1
                violations_log.append({
                    "rule_id": "R007",
                    "rule_name": "min_nights_le_max_nights",
                    "row_index": int(idx),
                    "listing_id": row_id,
                    "column": "minimum_nights,maximum_nights",
                    "value": f"minimum_nights={row.get('minimum_nights')}, maximum_nights={row.get('maximum_nights')}",
                    "details": f"Cross-field violation: minimum_nights ({row.get('minimum_nights')}) > maximum_nights ({row.get('maximum_nights')}) or missing boundary."
                })

        per_rule_summary = {}
        rule_descriptions = {
            "price_greater_than_zero": "price must be numeric and > 0",
            "minimum_nights_in_range": "minimum_nights must be in range [1, 365]",
            "availability_365_in_range": "availability_365 must be in range [0, 365]",
            "room_type_valid": "room_type must belong to allowed categorical set",
            "neighbourhood_group_valid": "neighbourhood_group_cleansed must be a valid borough/ward for the city",
            "coordinate_bounding_box": "latitude and longitude must fall within the city real bounding box",
            "min_nights_le_max_nights": "maximum_nights must be greater than or equal to minimum_nights"
        }
        rule_categories = {
            "price_greater_than_zero": "numeric_range",
            "minimum_nights_in_range": "numeric_range",
            "availability_365_in_range": "numeric_range",
            "room_type_valid": "categorical_consistency",
            "neighbourhood_group_valid": "categorical_consistency",
            "coordinate_bounding_box": "coordinate_sanity",
            "min_nights_le_max_nights": "cross_field_consistency"
        }

        for rule_name, count in per_rule_counts.items():
            per_rule_summary[rule_name] = {
                "rule_category": rule_categories[rule_name],
                "description": rule_descriptions[rule_name],
                "violation_count": count,
                "violation_pct": float(round((count / total_rows) * 100, 2)) if total_rows > 0 else 0.0
            }

        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "city": self.city_name,
            "bounding_box": self.city_bbox,
            "total_rows_validated": total_rows,
            "total_rules_evaluated": len(per_rule_counts),
            "total_violations_found": len(violations_log),
            "per_rule_summary": per_rule_summary,
            "rule_violations": violations_log
        }

        return report, violations_log

def validate_rules(dataset_path: str = "cleaned_data.csv", output_json_path: str = "rule_violations.json") -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    df = pd.read_csv(dataset_path, low_memory=False)
    validator = RuleValidator()
    report, _ = validator.validate_dataframe(df, dataset_path=dataset_path)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also save as validation_report.json for Module 3 specification
    with open("validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Successfully generated rule violations report '{output_json_path}' and 'validation_report.json'.")
    print(f"Total rows validated: {report['total_rows_validated']}")
    print(f"Total rule violations found: {report['total_violations_found']}")
    for r_name, summary in report["per_rule_summary"].items():
        print(f"  - {r_name}: {summary['violation_count']} violations ({summary['violation_pct']}%)")

    return report

if __name__ == "__main__":
    validate_rules()
