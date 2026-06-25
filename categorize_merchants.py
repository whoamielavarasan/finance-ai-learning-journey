import pandas as pd
import re

# ── Merchant Classification Rules ───────────────────────────────────────────
# Order matters: more specific patterns are checked before general ones.
# Each rule maps to (category, spend_type).

RULES = [
    # ── Groceries -- Necessary ──────────────────────────────────────────────
    (r"COSTCO WHSE",                         "Groceries",  "Necessary"),
    (r"SAFEWAY(?! FUEL)",                     "Groceries",  "Necessary"),
    (r"WAL-?MART(?! \w+ FUEL)|WM SUPERCENTER","Groceries",  "Necessary"),
    (r"FRED-MEYER(?! FUEL)|FRED M FUEL",       "Groceries",  "Necessary"),
    (r"SPICE WORLD|INDIA SUPERMARKET|INDIA METRO HYPER|APNA BAZAR|"
     r"BHARAT RATAN|SWAGATH HOME FOODS|ASIAN FAMILY MARKET|"
     r"INTERNATIONAL FOOD BAZAAR|MAYURI FOODS|PUYALLUP GROCERY",
                                               "Groceries",  "Necessary"),
    (r"TARGET",                               "Groceries",  "Necessary"),

    # ── Gas / Fuel -- Necessary ──────────────────────────────────────────────
    (r"COSTCO GAS|SAFEWAY FUEL|FRED M FUEL",  "Gas",        "Necessary"),

    # ── Restaurants / Fast Food -- Avoidable ────────────────────────────────
    (r"MCDONALD|BURGER KING|KFC|WINGSTOP|DOMINOS|SUBWAY|JOLLIBEE|"
     r"MOD PIZZA|PY \*PUYALLUP PIZZA|TANDOORI GRILL|SIZZLIN SPICE|"
     r"BASKIN|SQ \*BIRYANI CORNER|SQ \*KWALITY ICE CREAMS|"
     r"SQ \*SOUTHERN SPICE|SQ \*MAYURI BAKERY",
                                               "Dining Out", "Avoidable"),

    # ── Online / General Shopping -- Avoidable ──────────────────────────────
    (r"AMAZON|AMZN",                          "Shopping",   "Avoidable"),
    (r"TEMU\.?COM|TEMU COM",                  "Shopping",   "Avoidable"),
    (r"WALMART\.COM",                         "Shopping",   "Avoidable"),
    (r"DOLLAR ?TREE|FIVE BELOW|DAISO",         "Shopping",   "Avoidable"),
    (r"ROSS STORES|KOHL'?S|JCPENNEY|BATH AND BODY WORKS|CARTER'?S",
                                               "Shopping",   "Avoidable"),
    (r"GROUPON",                               "Shopping",   "Avoidable"),
    (r"WALGREENS|AMERICAS BEST",               "Shopping",   "Avoidable"),
    (r"7-ELEVEN",                               "Shopping",   "Avoidable"),
    (r"IKEA",                                  "Shopping",   "Avoidable"),

    # ── Subscriptions / Streaming -- Avoidable ──────────────────────────────
    (r"DISNEY PLUS|HLU\*HULUPLUS|HULU |CLAUDE\.AI SUBSCRIPTION",
                                               "Subscriptions", "Avoidable"),

    # ── Utilities -- Necessary ──────────────────────────────────────────────
    (r"PUGET SOUND ENERGY|QUANTUM FIBER|XFINITY MOBILE|TMOBILE|"
     r"WCI\*MURRYES DISPOSAL|PIERCE CO UTILITIES",
                                               "Utilities",  "Necessary"),

    # ── Insurance -- Necessary ──────────────────────────────────────────────
    (r"GEICO|STATE FARM|ALLSTATE",             "Insurance",  "Necessary"),

    # ── Medical / Health -- Necessary ───────────────────────────────────────
    (r"MED\*MULTICARE|RAINIER ANESTHESIA|ELEVATE-?PT BILLING|"
     r"PT BILLING|TACOMA PIERCE COUNTY HLT",
                                               "Medical",    "Necessary"),

    # ── Housing / HOA -- Necessary ──────────────────────────────────────────
    (r"FS PAY-?HOA ASSESSMENTS|FIRGROVE MUTUAL",
                                               "Housing",    "Necessary"),

    # ── Auto -- Necessary ────────────────────────────────────────────────────
    (r"FIRESTONE|GREGS JAPANESE AUTO|KORUM HYUNDAI|WA VEHICLE LICENSING",
                                               "Auto",       "Necessary"),

    # ── Parking / Travel -- Avoidable (discretionary outings) ──────────────
    (r"PARKING|PARKINGKITTY|WASHINGTON PARK MOBILE|MOUNT RAINIER NATL PARK|"
     r"PORTLAND JAPANESE GARD|SKAGIT VALLEY TUL",
                                               "Entertainment", "Avoidable"),

    # ── Shipping / Office -- Necessary ──────────────────────────────────────
    (r"FEDEX|USPS",                            "Shipping",   "Necessary"),

    # ── Personal Care -- Avoidable ───────────────────────────────────────────
    (r"DOLLY'?S EYEBROW THREADING",            "Personal Care", "Avoidable"),

    # ── Government / Legal -- Necessary ──────────────────────────────────────
    (r"PIERCE COUNTY AUDITOR|MINISTRYOFEXTERNALAFFA",
                                               "Government", "Necessary"),

    # ── Tech services -- Necessary (business/professional tool) ────────────
    (r"SQ \*IS TECH INNOVATIONLL",             "Professional Services", "Necessary"),

    # ── Refund services / Miscellaneous -- confirmed by user ────────────────
    (r"EAZY REFUND LLC",                       "Tax Preparation", "Necessary"),
    (r"PUYALLUP DI\b",                          "Shopping",        "Avoidable"),  # Deseret Industries (thrift store)
]

COMPILED_RULES = [(re.compile(pattern, re.IGNORECASE), category, spend_type)
                   for pattern, category, spend_type in RULES]


def classify_merchant(description: str) -> dict:
    for pattern, category, spend_type in COMPILED_RULES:
        if pattern.search(description):
            return {"category": category, "spend_type": spend_type}
    return {"category": "Unclassified", "spend_type": "Uncategorized"}


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    classifications = df["description"].apply(classify_merchant).apply(pd.Series)
    return pd.concat([df, classifications], axis=1)


if __name__ == "__main__":
    bofa = pd.read_csv("bofa_cc_expenses.csv")
    citi = pd.read_csv("citi_cc_expenses.csv")

    combined = pd.concat([bofa, citi], ignore_index=True)
    combined = categorize_dataframe(combined)

    print(f"Total rows: {len(combined)}")
    print(f"\n=== Category breakdown ===")
    print(combined.groupby(["category", "spend_type"])["amount"].agg(["sum", "count"])
          .sort_values("sum", ascending=False).to_string())

    unclassified = combined[combined["category"] == "Unclassified"]
    print(f"\n=== Unclassified rows: {len(unclassified)} ===")
    if len(unclassified) > 0:
        print(unclassified[["date", "description", "amount", "account"]].to_string())

    uncategorized_flagged = combined[combined["spend_type"] == "Uncategorized"]
    print(f"\n=== Flagged for manual review (matched but spend_type unclear): {len(uncategorized_flagged)} ===")
    if len(uncategorized_flagged) > 0:
        print(uncategorized_flagged[["date", "description", "amount", "category", "account"]].to_string())

    combined.to_csv("cc_expenses_categorized.csv", index=False)
    print(f"\n[INFO] Saved: cc_expenses_categorized.csv")
