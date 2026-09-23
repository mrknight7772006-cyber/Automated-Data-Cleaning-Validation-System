import os
import json
import datetime
import pandas as pd
from difflib import SequenceMatcher

class DeduplicationEngine:
    """
    Engine for detecting and removing exact and near-duplicate listings from the same host at the same address.
    Uses exact key matching and fuzzy string matching on listing names.
    """

    def __init__(self, similarity_threshold=0.50, coord_precision=4):
        self.similarity_threshold = similarity_threshold
        self.coord_precision = coord_precision

    @staticmethod
    def _clean_str(s):
        return str(s).lower().strip()

    def _calc_similarity(self, s1, s2):
        return SequenceMatcher(None, self._clean_str(s1), self._clean_str(s2)).ratio()

    def run_deduplication(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df_work = df.copy()
        before_rows = len(df_work)

        exact_removed_indices = set()
        fuzzy_removed_indices = set()
        flagged_records = []

        # 1. Exact full row duplicates
        full_dup_mask = df_work.duplicated(keep="first")
        for idx in df_work[full_dup_mask].index:
            exact_removed_indices.add(idx)
            first_idx = df_work[df_work.duplicated(keep="first") == False].index[0]
            flagged_records.append({
                "type": "exact_full_row",
                "host_id": df_work.loc[idx].get("host_id"),
                "latitude": df_work.loc[idx].get("latitude"),
                "longitude": df_work.loc[idx].get("longitude"),
                "retained_id": df_work.loc[first_idx].get("id"),
                "removed_id": df_work.loc[idx].get("id"),
                "retained_name": df_work.loc[first_idx].get("name"),
                "removed_name": df_work.loc[idx].get("name"),
                "similarity": 1.0
            })

        # Create coordinate rounding columns for grouping near addresses
        lat_col = f"_lat_round_{self.coord_precision}"
        lon_col = f"_lon_round_{self.coord_precision}"
        df_work[lat_col] = pd.to_numeric(df_work["latitude"], errors="coerce").round(self.coord_precision)
        df_work[lon_col] = pd.to_numeric(df_work["longitude"], errors="coerce").round(self.coord_precision)

        # 2. Group by host_id and location coordinates
        grouped = df_work.groupby(["host_id", lat_col, lon_col])

        for (h_id, lat, lon), group in grouped:
            if len(group) > 1:
                indices = group.index.tolist()
                for i in range(len(indices)):
                    idx1 = indices[i]
                    if idx1 in exact_removed_indices or idx1 in fuzzy_removed_indices:
                        continue

                    for j in range(i + 1, len(indices)):
                        idx2 = indices[j]
                        if idx2 in exact_removed_indices or idx2 in fuzzy_removed_indices:
                            continue

                        r1 = df_work.loc[idx1]
                        r2 = df_work.loc[idx2]

                        name1 = str(r1.get("name", ""))
                        name2 = str(r2.get("name", ""))
                        id1 = r1.get("id")
                        id2 = r2.get("id")

                        sim_score = self._calc_similarity(name1, name2)
                        is_exact_id = pd.notna(id1) and pd.notna(id2) and (id1 == id2)
                        is_exact_name = self._clean_str(name1) == self._clean_str(name2)

                        if is_exact_id or is_exact_name:
                            exact_removed_indices.add(idx2)
                            flagged_records.append({
                                "type": "exact",
                                "host_id": h_id,
                                "latitude": lat,
                                "longitude": lon,
                                "retained_id": id1,
                                "removed_id": id2,
                                "retained_name": name1,
                                "removed_name": name2,
                                "similarity": float(sim_score)
                            })
                        elif sim_score >= self.similarity_threshold:
                            fuzzy_removed_indices.add(idx2)
                            flagged_records.append({
                                "type": "fuzzy",
                                "host_id": h_id,
                                "latitude": lat,
                                "longitude": lon,
                                "retained_id": id1,
                                "removed_id": id2,
                                "retained_name": name1,
                                "removed_name": name2,
                                "similarity": float(sim_score)
                            })

        # Remove temporary rounded coordinate columns
        df_work = df_work.drop(columns=[lat_col, lon_col])

        all_removed_indices = exact_removed_indices.union(fuzzy_removed_indices)
        df_clean = df_work.drop(index=list(all_removed_indices)).reset_index(drop=True)
        after_rows = len(df_clean)

        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "before_row_count": before_rows,
            "after_row_count": after_rows,
            "exact_duplicates_count": len(exact_removed_indices),
            "fuzzy_duplicates_count": len(fuzzy_removed_indices),
            "total_duplicates_removed": len(all_removed_indices),
            "deduplication_summary": (
                f"Processed {before_rows} rows. Removed {len(exact_removed_indices)} exact duplicates "
                f"and {len(fuzzy_removed_indices)} fuzzy near-duplicates from the same host at the same address. "
                f"Final dataset has {after_rows} rows."
            ),
            "flagged_duplicates": flagged_records
        }

        return df_clean, report

def deduplicate_dataframe(df: pd.DataFrame, similarity_threshold=0.50) -> tuple[pd.DataFrame, dict]:
    engine = DeduplicationEngine(similarity_threshold=similarity_threshold)
    return engine.run_deduplication(df)
