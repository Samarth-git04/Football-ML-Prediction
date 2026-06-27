from __future__ import annotations

from pathlib import Path
from time import sleep
from io import StringIO
import random
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# If this file is inside data_gathering_scripts/, project root is one level above.
BASE_URL = "https://fifaindex.com/players"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT /"data"/"raw_data"/"fifaindex" 
CSV_DIR = OUTPUT_DIR / "csv"
PARQUET_DIR = OUTPUT_DIR / "parquet"

POSITION_GROUPS = {
    "gk": "goalkeeper",
    "def": "defender",
    "mid": "midfielder",
    "att": "attacker",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) "
        "Gecko/20100101 Firefox/127.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://fifaindex.com/players/",
    "Connection": "keep-alive",
}

REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.5
MAX_RETRIES = 5
MAX_PAGES_PER_POSITION = 600


def make_dirs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def sleep_politely() -> None:
    sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))


def fetch_page(session: requests.Session, position_code: str, page: int) -> str:
    params = {
        "page": page,
        "pos": position_code,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=30,
            )

            if response.status_code == 403:
                wait = 5 * attempt
                print(
                    f"403 Forbidden for pos={position_code}, page={page}. "
                    f"Waiting {wait}s then retrying..."
                )
                sleep(wait)
                continue

            response.raise_for_status()
            return response.text

        except Exception as exc:
            last_error = exc
            wait = 3 * attempt
            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed for "
                f"pos={position_code}, page={page}: {exc}. Waiting {wait}s..."
            )
            sleep(wait)

    raise RuntimeError(
        f"Failed to fetch pos={position_code}, page={page}"
    ) from last_error


def clean_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col


def extract_player_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))

    candidate_tables = []

    for table in tables:
        cols = [str(c).lower() for c in table.columns]
        joined = " ".join(cols)

        if "player" in joined and "ovr" in joined and "pot" in joined:
            candidate_tables.append(table)

    if not candidate_tables:
        return pd.DataFrame()

    df = candidate_tables[0].copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    # Remove empty/unnamed columns.
    df = df.loc[:, [str(col).strip() != "" and not str(col).lower().startswith("unnamed") for col in df.columns]]
    df = df.loc[:, df.columns != ""]

    return df


def has_next_page(html: str, current_page: int) -> bool:
    """
    Checks whether the HTML contains a link to the next page.
    """
    soup = BeautifulSoup(html, "lxml")
    next_page = current_page + 1

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if f"page={next_page}" in href:
            return True

    return False


def page_signature(df: pd.DataFrame) -> tuple:
    """
    Used to detect repeated pages.
    Some websites return page 1 again when page number is invalid.
    """
    if df.empty:
        return tuple()

    useful_cols = [col for col in ["player", "position", "nation", "team", "ovr", "pot"] if col in df.columns]

    if not useful_cols:
        return tuple(df.head(10).astype(str).agg("|".join, axis=1).tolist())

    return tuple(df[useful_cols].head(10).astype(str).agg("|".join, axis=1).tolist())


def scrape_position_group(
    session: requests.Session,
    position_code: str,
    position_group: str,
) -> pd.DataFrame:
    print(f"\nScraping position group: {position_group} ({position_code})")

    all_pages = []
    seen_signatures = set()

    for page in tqdm(range(1, MAX_PAGES_PER_POSITION + 1), desc=f"{position_group} pages"):
        html = fetch_page(session, position_code=position_code, page=page)
        df = extract_player_table(html)

        if df.empty:
            print(f"\nNo player table found for {position_group}, page {page}. Stopping.")
            break

        sig = page_signature(df)

        if sig in seen_signatures:
            print(f"\nRepeated page detected for {position_group}, page {page}. Stopping.")
            break

        seen_signatures.add(sig)

        df["source_position_code"] = position_code
        df["source_position_group"] = position_group
        df["source_page"] = page

        all_pages.append(df)

        next_exists = has_next_page(html, page)

        if not next_exists:
            print(f"\nNo next page found for {position_group} after page {page}. Stopping.")
            break

        sleep_politely()

    if not all_pages:
        return pd.DataFrame()

    result = pd.concat(all_pages, ignore_index=True)
    return result


def standardise_player_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    df["scrape_source"] = "fifaindex"
    df["game_version"] = "FC 26"

    return df


def main() -> None:
    make_dirs()

    print(f"Saving FIFA Index data to: {OUTPUT_DIR}")

    session = requests.Session()

    all_groups = []

    for position_code, position_group in POSITION_GROUPS.items():
        group_df = scrape_position_group(
            session=session,
            position_code=position_code,
            position_group=position_group,
        )

        if group_df.empty:
            print(f"No data scraped for {position_group}")
            continue

        group_df = standardise_player_df(group_df)

        csv_path = CSV_DIR / f"fifaindex_fc26_{position_code}_{position_group}.csv"
        parquet_path = PARQUET_DIR / f"fifaindex_fc26_{position_code}_{position_group}.parquet"

        group_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved CSV: {csv_path}")

        try:
            group_df.to_parquet(parquet_path, index=False)
            print(f"Saved Parquet: {parquet_path}")
        except Exception as exc:
            print(f"Could not save Parquet for {position_group}: {exc}")

        print(f"{position_group} rows scraped: {len(group_df):,}")

        all_groups.append(group_df)

        sleep_politely()

    if not all_groups:
        raise RuntimeError("No FIFA Index data scraped.")

    all_players = pd.concat(all_groups, ignore_index=True)
    all_players = all_players.drop_duplicates()

    all_csv_path = CSV_DIR / "fifaindex_fc26_all_position_groups.csv"
    all_parquet_path = PARQUET_DIR / "fifaindex_fc26_all_position_groups.parquet"

    all_players.to_csv(all_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved combined CSV: {all_csv_path}")

    try:
        all_players.to_parquet(all_parquet_path, index=False)
        print(f"Saved combined Parquet: {all_parquet_path}")
    except Exception as exc:
        print(f"Could not save combined Parquet: {exc}")

    print("\nDone.")
    print(f"Total rows scraped before final modelling cleanup: {len(all_players):,}")
    print("Columns:")
    print(all_players.columns.tolist())


if __name__ == "__main__":
    main()