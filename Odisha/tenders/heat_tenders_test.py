#!/usr/bin/env python3
"""
EDA + heuristic tagging for heat / heatwave-related tenders in Odisha.

Assumes a CSV with at least:
- 'tender_title'
- 'Work Description'

Optional but used if present:
- 'Product Category', 'Tender Category', 'Department', 'location'
"""

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# ---------- CONFIG ----------

INPUT_CSV = "/Users/saurabhlevin/Deployment/Heat_EDA/Odisha/tenders/odisha_all_tenders.csv"
OUTPUT_TAGGED_CSV = "/Users/saurabhlevin/Deployment/Heat_EDA/Odisha/tenders/odisha_all_tenders_heat_tagged.csv"
OUTPUT_HEAT_ONLY_CSV = "/Users/saurabhlevin/Deployment/Heat_EDA/Odisha/tenders/odisha_heat_related_tenders.csv"

TEXT_COLS = ["tender_title", "Work Description"]

# Keyword groups for heuristic tagging
EXPLICIT_HEAT_KEYWORDS = [
    r"heatwave", r"heat wave", r"heat-wave",
    r"heat stress", r"heat stroke",
    r"heat action plan", r"heat resilience",
    r"extreme heat", r"high temperature", r"rising temperature",
]

DRINKING_WATER_KEYWORDS = [
    r"drinking water", r"water kiosk", r"water atm",
    r"hydration point", r"hydration centre", r"cold water",
    r"tube well", r"borewell", r"hand pump",
    r"overhead tank", r"oh tank", r"stand post",
]

GREENING_SHADE_KEYWORDS = [
    r"tree plantation", r"avenue plantation", r"plantation",
    r"green belt", r"urban forestry", r"urban forest",
    r"park development", r"shade tree", r"shade net",
    r"shelter belt",
]

ROOF_COOLING_KEYWORDS = [
    r"cool roof", r"white roof", r"white topping",
    r"roof insulation", r"thermal insulation",
    r"heat reflective", r"reflective coating",
]

SHELTER_INFRA_KEYWORDS = [
    r"shelter", r"community hall", r"relief centre",
    r"night shelter", r"transit shelter", r"rest shed",
    r"waiting shed", r"bus stand shelter",
]

HEALTH_RESPONSE_KEYWORDS = [
    r"health centre", r"health center",
    r"primary health centre", r"phc", r"chc",
    r"hospital", r"clinic", r"ambulance",
    r"emergency response", r"first aid",
]

OTHER_RELEVANT_KEYWORDS = [
    r"heat resistant", r"climate resilient",
    r"cooling centre", r"cooling center",
    r"shade", r"canopy", r"awning"
]

# Seed descriptions for TF-IDF similarity scoring
HEAT_SEED_TEXTS = [
    "Construction and operation of cooling centers and shelters during extreme heat and heatwaves.",
    "Provision of safe drinking water points, water kiosks and overhead tanks to reduce heat stress.",
    "Tree plantation and urban greening to reduce urban heat island effect and provide shade.",
    "Cool roof or reflective roof interventions to reduce indoor temperatures.",
    "Health system preparedness for heatwaves, including awareness, emergency response and infrastructure."
]

# Cosine similarity threshold for "semantically similar to heatwave work"
HEAT_SIM_THRESHOLD = 0.15  # tune this after you inspect results


# ---------- HELPERS ----------

def safe_lower(x):
    if isinstance(x, str):
        return x.lower()
    return ""

def build_text_field(df, cols):
    texts = []
    for _, row in df[cols].iterrows():
        parts = [str(row[c]) for c in cols if pd.notna(row[c])]
        texts.append(" | ".join(parts))
    return texts

def keyword_flag(series, patterns):
    """
    Returns a boolean Series indicating presence of *any* pattern.
    Patterns are regular expressions (assumed case-insensitive).
    """
    if series.isna().all():
        return pd.Series(False, index=series.index)

    # use a non-capturing group to avoid pandas "has match groups" warnings
    joined_pattern = "(?:" + "|".join(patterns) + ")"
    return series.str.contains(joined_pattern, case=False, regex=True, na=False)


# ---------- MAIN PIPELINE ----------

def main():
    # 1. Load data
    df = pd.read_csv(INPUT_CSV)

    print("Basic info:")
    print(df.shape)
    print(df.columns)

    # quick NA overview
    print("\nMissing values per column:")
    print(df.isna().sum().sort_values(ascending=False))

    # 2. Build combined text field
    for col in TEXT_COLS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    df["combined_text"] = build_text_field(df, TEXT_COLS)
    df["combined_text_clean"] = df["combined_text"].apply(safe_lower)

    # 3. Rule-based flags
    df["flag_explicit_heat"] = keyword_flag(df["combined_text_clean"], EXPLICIT_HEAT_KEYWORDS)
    df["flag_drinking_water"] = keyword_flag(df["combined_text_clean"], DRINKING_WATER_KEYWORDS)
    df["flag_greening_shade"] = keyword_flag(df["combined_text_clean"], GREENING_SHADE_KEYWORDS)
    df["flag_roof_cooling"] = keyword_flag(df["combined_text_clean"], ROOF_COOLING_KEYWORDS)
    df["flag_shelter_infra"] = keyword_flag(df["combined_text_clean"], SHELTER_INFRA_KEYWORDS)
    df["flag_health_response"] = keyword_flag(df["combined_text_clean"], HEALTH_RESPONSE_KEYWORDS)
    df["flag_other_heat_related"] = keyword_flag(df["combined_text_clean"], OTHER_RELEVANT_KEYWORDS)

    # aggregate rule-based flag
    rule_flags = [
        "flag_explicit_heat",
        "flag_drinking_water",
        "flag_greening_shade",
        "flag_roof_cooling",
        "flag_shelter_infra",
        "flag_health_response",
        "flag_other_heat_related",
    ]
    df["flag_any_rule_based_heat"] = df[rule_flags].any(axis=1)

    # 4. TF-IDF similarity to heatwave seed descriptions
    # Build corpus: all tender texts + seed texts
    corpus = list(df["combined_text_clean"].fillna("")) + HEAT_SEED_TEXTS

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # last N rows are seeds
    n_docs = df.shape[0]
    # convert to CSR explicitly (avoid attribute access issues on the base spmatrix type)
    tfidf_matrix = csr_matrix(tfidf_matrix)
    seed_vecs = tfidf_matrix[n_docs:, :]
    tender_vecs = tfidf_matrix[:n_docs, :]

    # cosine similarity between each tender and mean of seeds
    # `seed_vecs.mean` returns a numpy.matrix which newer numpy/scikit-learn
    # do not accept directly; convert to a plain ndarray first.
    seed_mean = np.asarray(seed_vecs.mean(axis=0))
    sim_scores = cosine_similarity(tender_vecs, seed_mean)

    # sim_scores is (n_docs, 1)
    df["heat_semantic_score"] = sim_scores.ravel()
    df["flag_heat_semantic"] = df["heat_semantic_score"] >= HEAT_SIM_THRESHOLD

    # 5. Combined "likely heat-related" flag
    df["flag_likely_heat_related"] = df["flag_any_rule_based_heat"] | df["flag_heat_semantic"]

    # 6. Basic summaries
    print("\nRule-based flag counts:")
    print(df[rule_flags + ["flag_any_rule_based_heat"]].sum().sort_values(ascending=False))

    print("\nSemantic similarity stats (heat_semantic_score):")
    print(df["heat_semantic_score"].describe())

    print("\nCounts of likely heat-related tenders:")
    print(df["flag_likely_heat_related"].value_counts())

    # 7. Inspect some examples
    print("\nSample explicit heatwave mentions:")
    print(
        df.loc[df["flag_explicit_heat"], ["tender_title", "Work Description", "heat_semantic_score"]]
        .head(20)
        .to_string(index=False)
    )

    print("\nSample high semantic-score tenders (top 20):")
    print(
        df.sort_values("heat_semantic_score", ascending=False)
          .loc[:, ["tender_title", "Work Description", "heat_semantic_score"]]
          .head(20)
          .to_string(index=False)
    )

    # 8. Export tagged dataset & heat-only subset
    df.to_csv(OUTPUT_TAGGED_CSV, index=False)

    df_heat = df[df["flag_likely_heat_related"]].copy()
    df_heat.to_csv(OUTPUT_HEAT_ONLY_CSV, index=False)

    print(f"\nSaved tagged dataset to: {OUTPUT_TAGGED_CSV}")
    print(f"Saved heat-related subset to: {OUTPUT_HEAT_ONLY_CSV}")


if __name__ == "__main__":
    main()
