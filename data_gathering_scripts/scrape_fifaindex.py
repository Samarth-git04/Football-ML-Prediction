# This script is long and can take up to a day if you want just the recent last 5 years of data then use top 5 in the GAME_VERSION
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


BASE_URL_TEMPLATE = "https://fifaindex.com/players/{version_slug}"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_data" / "fifaindex"
CSV_DIR = OUTPUT_DIR / "csv"
PARQUET_DIR = OUTPUT_DIR / "parquet"

POSITION_GROUPS = {
    "gk": "goalkeeper",
    "def": "defender",
    "mid": "midfielder",
    "att": "attacker",
}

# This scrapes FC 26 back to FIFA 10.
GAME_VERSIONS = {
    "fc26": "FC 26",
    "fc25": "FC 25",
    "fc24": "FC 24",
    "fifa23": "FIFA 23",
    "fifa22": "FIFA 22",
    "fifa21": "FIFA 21",
    "fifa20": "FIFA 20",
    "fifa19": "FIFA 19",
    "fifa18": "FIFA 18",
    "fifa17": "FIFA 17",
    "fifa16": "FIFA 16",
    "fifa15": "FIFA 15",
    "fifa14": "FIFA 14",
    "fifa13": "FIFA 13",
    "fifa12": "FIFA 12",
    "fifa11": "FIFA 11",
    "fifa10": "FIFA 10",
    "fifa09": "FIFA 09",
    "fifa08": "FIFA 08",
    "fifa07": "FIFA 07",
    "fifa06": "FIFA 06",
    "fifa05": "FIFA 05"
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


def fetch_page(
    session: requests.Session,
    version_slug: str,
    position_code: str,
    page: int,
) -> str:
    url = BASE_URL_TEMPLATE.format(version_slug=version_slug)

    params = {
        "page": page,
        "pos": position_code,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=30,
            )

            if response.status_code == 403:
                wait = 10 * attempt
                print(
                    f"403 Forbidden for version={version_slug}, "
                    f"pos={position_code}, page={page}. "
                    f"Waiting {wait}s then retrying..."
                )
                sleep(wait)
                continue

            response.raise_for_status()
            return response.text

        except Exception as exc:
            last_error = exc
            wait = 5 * attempt
            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed for "
                f"version={version_slug}, pos={position_code}, page={page}: {exc}. "
                f"Waiting {wait}s..."
            )
            sleep(wait)

    raise RuntimeError(
        f"Failed to fetch version={version_slug}, pos={position_code}, page={page}"
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

    df = df.loc[
        :,
        [
            str(col).strip() != ""
            and not str(col).lower().startswith("unnamed")
            for col in df.columns
        ],
    ]

    df = df.loc[:, df.columns != ""]

    return df


def has_next_page(html: str, current_page: int) -> bool:
    soup = BeautifulSoup(html, "lxml")
    next_page = current_page + 1

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if f"page={next_page}" in href:
            return True

    return False


def page_signature(df: pd.DataFrame) -> tuple:
    if df.empty:
        return tuple()

    useful_cols = [
        col
        for col in ["player", "position", "nation", "team", "ovr", "pot"]
        if col in df.columns
    ]

    if not useful_cols:
        return tuple(df.head(10).astype(str).agg("|".join, axis=1).tolist())

    return tuple(
        df[useful_cols]
        .head(10)
        .astype(str)
        .agg("|".join, axis=1)
        .tolist()
    )


def scrape_position_group(
    session: requests.Session,
    version_slug: str,
    game_version: str,
    position_code: str,
    position_group: str,
) -> pd.DataFrame:
    print(
        f"\nScraping {game_version} | "
        f"{position_group} ({position_code})"
    )

    all_pages = []
    seen_signatures = set()

    for page in tqdm(
        range(1, MAX_PAGES_PER_POSITION + 1),
        desc=f"{version_slug}-{position_group} pages",
    ):
        html = fetch_page(
            session=session,
            version_slug=version_slug,
            position_code=position_code,
            page=page,
        )

        df = extract_player_table(html)

        if df.empty:
            print(
                f"\nNo player table found for {game_version}, "
                f"{position_group}, page {page}. Stopping."
            )
            break

        sig = page_signature(df)

        if sig in seen_signatures:
            print(
                f"\nRepeated page detected for {game_version}, "
                f"{position_group}, page {page}. Stopping."
            )
            break

        seen_signatures.add(sig)

        df["source_position_code"] = position_code
        df["source_position_group"] = position_group
        df["source_page"] = page
        df["scrape_source"] = "fifaindex"
        df["game_version"] = game_version
        df["game_version_slug"] = version_slug

        all_pages.append(df)

        next_exists = has_next_page(html, page)

        if not next_exists:
            print(
                f"\nNo next page found for {game_version}, "
                f"{position_group} after page {page}. Stopping."
            )
            break

        sleep_politely()

    if not all_pages:
        return pd.DataFrame()

    result = pd.concat(all_pages, ignore_index=True)
    result = result.drop_duplicates()

    return result


def save_df(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {csv_path}")

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet: {parquet_path}")
    except Exception as exc:
        print(f"Could not save Parquet: {exc}")


def scrape_game_version(
    session: requests.Session,
    version_slug: str,
    game_version: str,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"STARTING VERSION: {game_version} ({version_slug})")
    print("=" * 80)

    version_groups = []

    for position_code, position_group in POSITION_GROUPS.items():
        csv_path = CSV_DIR / f"fifaindex_{version_slug}_{position_code}_{position_group}.csv"
        parquet_path = PARQUET_DIR / f"fifaindex_{version_slug}_{position_code}_{position_group}.parquet"

        # Resume-friendly behaviour:
        # if the group CSV already exists, load it instead of scraping again.
        if csv_path.exists():
            print(f"\nExisting file found, loading instead of scraping: {csv_path}")
            group_df = pd.read_csv(csv_path)
            version_groups.append(group_df)
            continue

        group_df = scrape_position_group(
            session=session,
            version_slug=version_slug,
            game_version=game_version,
            position_code=position_code,
            position_group=position_group,
        )

        if group_df.empty:
            print(f"No data scraped for {game_version} {position_group}")
            continue

        save_df(group_df, csv_path, parquet_path)

        print(
            f"{game_version} {position_group} rows scraped: "
            f"{len(group_df):,}"
        )

        version_groups.append(group_df)

        sleep_politely()

    if not version_groups:
        print(f"No data scraped for {game_version}")
        return pd.DataFrame()

    version_df = pd.concat(version_groups, ignore_index=True)
    version_df = version_df.drop_duplicates()

    all_csv_path = CSV_DIR / f"fifaindex_{version_slug}_all_position_groups.csv"
    all_parquet_path = PARQUET_DIR / f"fifaindex_{version_slug}_all_position_groups.parquet"

    save_df(version_df, all_csv_path, all_parquet_path)

    print(
        f"\nFinished {game_version}. "
        f"Total rows before modelling cleanup: {len(version_df):,}"
    )

    return version_df


def main() -> None:
    make_dirs()

    print(f"Saving FIFA Index data to: {OUTPUT_DIR}")

    session = requests.Session()

    all_versions = []

    for version_slug, game_version in GAME_VERSIONS.items():
        version_df = scrape_game_version(
            session=session,
            version_slug=version_slug,
            game_version=game_version,
        )

        if not version_df.empty:
            all_versions.append(version_df)

        # Extra pause between game versions.
        sleep_politely()

    if not all_versions:
        raise RuntimeError("No FIFA Index data scraped.")

    all_players = pd.concat(all_versions, ignore_index=True)
    all_players = all_players.drop_duplicates()

    all_versions_csv_path = CSV_DIR / "fifaindex_fc25_to_fifa10_all_versions.csv"
    all_versions_parquet_path = PARQUET_DIR / "fifaindex_fc25_to_fifa10_all_versions.parquet"

    save_df(all_players, all_versions_csv_path, all_versions_parquet_path)

    print("\nDone.")
    print(f"Total rows scraped across all versions: {len(all_players):,}")
    print("Columns:")
    print(all_players.columns.tolist())


if __name__ == "__main__":
    main()