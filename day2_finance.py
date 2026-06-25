import pandas as pd


def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw rows loaded: {len(df)}")

    df = parse_dates(df)
    df = parse_amounts(df)
    df = normalize_text(df)
    df = drop_invalid_rows(df)
    df = drop_exact_duplicates(df)
    df = reset_index(df)

    print(f"[INFO] Clean rows after pipeline: {len(df)}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    .assign() returns a NEW DataFrame with 'date' replaced.
    Pandas 3.x Copy-on-Write makes this the correct pattern.
    df["date"] = ... would mutate in-place — unreliable under CoW.
    """
    return df.assign(date=pd.to_datetime(df["date"], errors="coerce"))


def parse_amounts(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(amount=pd.to_numeric(df["amount"], errors="coerce"))


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    .assign() can take multiple keyword arguments — each is a new/replaced column.
    Both description and category are normalized in a single pass.
    """
    return df.assign(
        description = df["description"].str.strip(),
        category    = df["category"].str.strip().str.title()
    )


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropna() already returns a new DataFrame — no .assign() needed here.
    """
    before  = len(df)
    df      = df.dropna(subset=["date", "amount", "category"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} invalid row(s) with NULL date/amount/category.")
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    drop_duplicates() also returns a new DataFrame — no .assign() needed.
    """
    before  = len(df)
    df      = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} exact duplicate row(s).")
    return df


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All five date features added in a single .assign() call.
    Each expression can reference the ORIGINAL df columns —
    they are computed independently, not sequentially.
    Note: is_weekend references df["date"], not the new day_of_week column
    being created in the same .assign() call — that's a Pandas limitation.
    So we chain a second .assign() for is_weekend.
    """
    return (
        df.assign(
            year        = df["date"].dt.year,
            month       = df["date"].dt.month,
            day         = df["date"].dt.day,
            day_of_week = df["date"].dt.dayofweek   # 0=Monday, 6=Sunday
        )
        .assign(
            is_weekend = lambda d: d["day_of_week"] >= 5
            # lambda here: d refers to the DataFrame AFTER the first .assign()
            # so day_of_week already exists when is_weekend is computed.
            # lambda = anonymous function — C# equivalent: Func<DataFrame, Series>
        )
    )


# --- ENTRY POINT ---
if __name__ == "__main__":

    df = load_and_clean("transactions.csv")
    df = extract_date_features(df)

    print("\n--- Final Clean DataFrame ---")
    print(df.to_string())       # to_string() prevents Pandas truncating wide output

    print("\n--- dtypes after full pipeline ---")
    print(df.dtypes)

    print("\n--- Null check (all zeros expected) ---")
    print(df.isnull().sum())