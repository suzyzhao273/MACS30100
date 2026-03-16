import os
import io
import json
import re
import math
from collections import defaultdict
from typing import Optional, Set

import pandas as pd
import zstandard as zstd
from tqdm import tqdm



# CONFIG

# Base table to enrich
BASE_TABLE_CSV = "base_song_table_2008_2018_enriched.csv"

# Reddit submission zst file
MUSIC_ZST = "reddit/subreddits25/Music_submissions.zst"

# Output directory
OUT_DIR = "reddit_artist_output_light_music"
os.makedirs(OUT_DIR, exist_ok=True)

# Output files
ENRICHED_OUTPUT_CSV = os.path.join(
    OUT_DIR, "base_song_table_2008_2018_enriched_with_artist_reddit_music.csv"
)
REDDIT_FEATURES_ONLY_CSV = os.path.join(
    OUT_DIR, "reddit_artist_features_music_only.csv"
)

# Shrunk zst output
WRITE_SHRUNK_FILES = True
MUSIC_SHRUNK_ZST = os.path.join(OUT_DIR, "music_submission_shrunk.zst")

# Use shrunk file for matching
USE_SHRUNK_FOR_MATCHING = True

# Time filter for Reddit posts
MIN_POST_DATE = pd.Timestamp("2007-12-25")
MAX_POST_DATE = pd.Timestamp("2019-01-03 23:59:59")

# Matching rules
MAX_TEXTS_PER_BUCKET = 30



# TEXT UTILS

def normalize_text(text) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return set(text.split())


def build_boundary_pattern(phrase: str) -> Optional[re.Pattern]:
    if not phrase:
        return None
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)


def phrase_hit(pattern: Optional[re.Pattern], text: str) -> bool:
    if pattern is None or not text:
        return False
    return pattern.search(text) is not None


def choose_anchor_token(phrase: str) -> Optional[str]:
    if not phrase:
        return None
    toks = phrase.split()
    if not toks:
        return None
    toks = sorted(toks, key=lambda x: (-len(x), x))
    for tok in toks:
        if len(tok) >= 4:
            return tok
    return toks[0]


def safe_full_text(title: str, selftext: str) -> str:
    title = "" if title is None else str(title)
    selftext = "" if selftext is None else str(selftext)

    if selftext.lower() in {"[deleted]", "[removed]", "nan", "none"}:
        selftext = ""

    return f"{title} {selftext}".strip()


def json_dumps_safe(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def safe_controversy(score: float, num_comments: float) -> float:
    """
    Stable controversy proxy.

    Reddit score can be small, zero, or even negative in some datasets.
    Using abs(score) avoids division-by-zero at score=-1 and prevents
    negative controversy values.
    """
    return num_comments / (abs(score) + 1.0)


# ZST STREAMING

def stream_zst_json(zst_path: str, desc: str = "stream"):
    file_size = os.path.getsize(zst_path)

    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

            with tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as pbar:
                last_pos = 0

                for line in text_stream:
                    current_pos = fh.tell()
                    if current_pos > last_pos:
                        pbar.update(current_pos - last_pos)
                        last_pos = current_pos

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

                if last_pos < file_size:
                    pbar.update(file_size - last_pos)


def write_zst_jsonl(records_iter, out_path: str, desc: str = "write_zst"):
    cctx = zstd.ZstdCompressor(level=6)
    with open(out_path, "wb") as fh:
        with cctx.stream_writer(fh) as writer:
            for rec in tqdm(records_iter, desc=desc):
                line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
                writer.write(line)



# STEP 1: SHRINK RAW REDDIT FILE

def shrink_reddit_zst(
    input_zst: str,
    output_zst: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp
):
    total_seen = 0
    kept = 0
    dropped_date = 0
    dropped_zero = 0
    dropped_bad_date = 0

    def filtered_records():
        nonlocal total_seen, kept, dropped_date, dropped_zero, dropped_bad_date

        for post in stream_zst_json(input_zst, desc=f"shrink:{os.path.basename(input_zst)}"):
            total_seen += 1

            created_utc = post.get("created_utc")
            if created_utc is None:
                dropped_bad_date += 1
                continue

            post_dt = pd.to_datetime(created_utc, unit="s", errors="coerce")
            if pd.isna(post_dt):
                dropped_bad_date += 1
                continue

            if not (min_date <= post_dt <= max_date):
                dropped_date += 1
                continue

            score = post.get("score", 0)
            num_comments = post.get("num_comments", 0)

            try:
                score = 0 if score is None else float(score)
            except Exception:
                score = 0

            try:
                num_comments = 0 if num_comments is None else float(num_comments)
            except Exception:
                num_comments = 0

            if score == 0 and num_comments == 0:
                dropped_zero += 1
                continue

            yield {
                "id": post.get("id"),
                "subreddit": post.get("subreddit"),
                "created_utc": created_utc,
                "score": score,
                "num_comments": num_comments,
                "title": post.get("title", ""),
                "selftext": post.get("selftext", "")
            }
            kept += 1

    write_zst_jsonl(filtered_records(), output_zst, desc=f"write:{os.path.basename(output_zst)}")

    summary = {
        "input_zst": input_zst,
        "output_zst": output_zst,
        "total_seen": total_seen,
        "kept": kept,
        "dropped_date": dropped_date,
        "dropped_zero": dropped_zero,
        "dropped_bad_date": dropped_bad_date
    }

    print("\n[SHRINK SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(output_zst + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



# READ BASE TABLE + PREP SONG FIELDS

def prepare_song_table_from_base(base_csv: str) -> pd.DataFrame:
    """
    Read the original base table and add only the fields needed for matching.
    This function does not overwrite the original file.

    Required logical fields:
    - song_id
    - artist
    - song
    - album
    - release_date
    """
    df = pd.read_csv(base_csv)
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}

    if "artist" not in df.columns:
        for cand in ["artist_name", "artist_names", "artists", "performer", "singer"]:
            if cand in df.columns:
                rename_map[cand] = "artist"
                break

    if "song" not in df.columns:
        for cand in ["song_name", "track_name", "track", "track_title", "title"]:
            if cand in df.columns:
                rename_map[cand] = "song"
                break

    if "album" not in df.columns:
        for cand in ["album_name", "record_name", "release_name"]:
            if cand in df.columns:
                rename_map[cand] = "album"
                break

    if "release_date" not in df.columns:
        for cand in [
            "track_release_date", "song_release_date", "date", "release",
            "release_day", "album_release_date"
        ]:
            if cand in df.columns:
                rename_map[cand] = "release_date"
                break

    df = df.rename(columns=rename_map)

    required = ["song_id", "artist", "song", "album", "release_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("\nAvailable columns in base table:")
        print(df.columns.tolist())
        raise ValueError(f"Missing required columns in base table: {missing}")

    df["artist"] = df["artist"].fillna("").astype(str)
    df["song"] = df["song"].fillna("").astype(str)
    df["album"] = df["album"].fillna("").astype(str)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df[df["release_date"].notna()].copy()

    df["baseline_start"] = df["release_date"] - pd.Timedelta(days=30)
    df["baseline_end"] = df["release_date"] - pd.Timedelta(days=7)

    df["release_start"] = df["release_date"] - pd.Timedelta(days=7)
    df["release_end"] = df["release_date"] + pd.Timedelta(days=7)

    df["artist_norm"] = df["artist"].apply(normalize_text)
    df["song_norm"] = df["song"].apply(normalize_text)
    df["artist_pattern"] = df["artist_norm"].apply(build_boundary_pattern)

    return df.reset_index(drop=True)



# FAST MATCHING INDEX

def build_matching_index(song_df: pd.DataFrame):
    artist_anchor_map = defaultdict(set)
    artist_meta_map = {}

    for artist_norm, g in song_df.groupby("artist_norm", dropna=False):
        if not artist_norm:
            continue

        artist_pattern = g["artist_pattern"].iloc[0]
        anchor = choose_anchor_token(artist_norm)
        if anchor:
            artist_anchor_map[anchor].add(artist_norm)

        songs = []
        for _, row in g.iterrows():
            songs.append({
                "song_id": row["song_id"],
                "artist": row["artist"],
                "song": row["song"],
                "album": row["album"],
                "release_date": row["release_date"],
                "baseline_start": row["baseline_start"],
                "baseline_end": row["baseline_end"],
                "release_start": row["release_start"],
                "release_end": row["release_end"],
            })

        artist_meta_map[artist_norm] = {
            "artist_pattern": artist_pattern,
            "songs": songs
        }

    return artist_anchor_map, artist_meta_map



# WINDOW HELPERS

def get_windows_for_song(post_dt: pd.Timestamp, song_row: dict):
    windows = []

    if pd.notna(song_row["baseline_start"]) and pd.notna(song_row["baseline_end"]):
        if song_row["baseline_start"] <= post_dt <= song_row["baseline_end"]:
            windows.append("baseline")

    if pd.notna(song_row["release_start"]) and pd.notna(song_row["release_end"]):
        if song_row["release_start"] <= post_dt <= song_row["release_end"]:
            windows.append("release")

    return windows



# AGGREGATION BUCKET

def init_song_window_bucket():
    return {
        "volume_weighted": 0.0,
        "controversy_mean_weighted_num": 0.0,
        "controversy_mean_weighted_den": 0.0,
        "artist_match_count_raw": 0,
        "artist_texts": []
    }


def maybe_append(lst: list, value: str, max_n: Optional[int] = None):
    if max_n is None or len(lst) < max_n:
        lst.append(value)


def finalize_bucket(bucket: dict) -> dict:
    return {
        "volume": bucket["volume_weighted"],
        "controversy_mean": (
            bucket["controversy_mean_weighted_num"] / bucket["controversy_mean_weighted_den"]
            if bucket["controversy_mean_weighted_den"] > 0 else 0.0
        ),
        "artist_match_count_raw": bucket["artist_match_count_raw"],
        "artist_text_list": json_dumps_safe(bucket["artist_texts"]),
        "artist_text_joined": " ||| ".join(bucket["artist_texts"]),
    }



# MATCH ONE POST (ARTIST ONLY)

def match_one_post(
    post_dt: pd.Timestamp,
    full_text_norm: str,
    tokens: Set[str],
    artist_anchor_map,
    artist_meta_map
):
    candidate_artists = set()
    for tok in tokens:
        if tok in artist_anchor_map:
            candidate_artists.update(artist_anchor_map[tok])

    matched_allocations = []

    for artist_norm in candidate_artists:
        artist_meta = artist_meta_map.get(artist_norm)
        if artist_meta is None:
            continue

        if not phrase_hit(artist_meta["artist_pattern"], full_text_norm):
            continue

        active_songs = []
        for s in artist_meta["songs"]:
            windows = get_windows_for_song(post_dt, s)
            if windows:
                s2 = dict(s)
                s2["windows"] = windows
                active_songs.append(s2)

        if not active_songs:
            continue

        n = len(active_songs)
        w = 1.0 / n
        for s in active_songs:
            matched_allocations.append({
                "song_id": s["song_id"],
                "windows": s["windows"],
                "weight": w
            })

    return matched_allocations



# PROCESS MUSIC SUBREDDIT

def process_music_subreddit(zst_path: str, song_df: pd.DataFrame):
    artist_anchor_map, artist_meta_map = build_matching_index(song_df)
    agg = defaultdict(init_song_window_bucket)

    processed = 0
    matched_posts = 0
    raw_allocations = 0

    for post in stream_zst_json(zst_path, desc="match:music"):
        processed += 1

        created_utc = post.get("created_utc")
        if created_utc is None:
            continue

        post_dt = pd.to_datetime(created_utc, unit="s", errors="coerce")
        if pd.isna(post_dt):
            continue

        title = post.get("title", "")
        selftext = post.get("selftext", "")
        full_text = safe_full_text(title, selftext)
        full_text_norm = normalize_text(full_text)

        if not full_text_norm:
            continue

        tokens = tokenize(full_text_norm)
        if not tokens:
            continue

        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)

        try:
            score = float(score) if score is not None else 0.0
        except Exception:
            score = 0.0

        try:
            num_comments = float(num_comments) if num_comments is not None else 0.0
        except Exception:
            num_comments = 0.0

        controversy_post = safe_controversy(score=score, num_comments=num_comments)

        allocations = match_one_post(
            post_dt=post_dt,
            full_text_norm=full_text_norm,
            tokens=tokens,
            artist_anchor_map=artist_anchor_map,
            artist_meta_map=artist_meta_map
        )

        if not allocations:
            continue

        matched_posts += 1
        raw_allocations += len(allocations)

        for alloc in allocations:
            sid = alloc["song_id"]
            weight = alloc["weight"]
            windows = alloc["windows"]

            for wname in windows:
                bucket = agg[(sid, wname)]
                bucket["volume_weighted"] += weight
                bucket["controversy_mean_weighted_num"] += controversy_post * weight
                bucket["controversy_mean_weighted_den"] += weight
                bucket["artist_match_count_raw"] += 1
                maybe_append(bucket["artist_texts"], full_text, MAX_TEXTS_PER_BUCKET)

    print(f"\n[music] processed={processed:,}")
    print(f"[music] matched_posts={matched_posts:,}")
    print(f"[music] raw_allocations={raw_allocations:,}")

    return agg



# AGG TO FEATURES TABLE

def agg_to_music_feature_df(song_df: pd.DataFrame, agg: dict) -> pd.DataFrame:
    out_rows = []

    for _, row in song_df.iterrows():
        sid = row["song_id"]
        b = finalize_bucket(agg.get((sid, "baseline"), init_song_window_bucket()))
        r = finalize_bucket(agg.get((sid, "release"), init_song_window_bucket()))

        out_rows.append({
            "song_id": sid,
            "music_volume_baseline": b["volume"],
            "music_volume_release": r["volume"],
            "music_controversy_baseline": b["controversy_mean"],
            "music_controversy_release": r["controversy_mean"],
            "music_artist_match_count_raw_baseline": b["artist_match_count_raw"],
            "music_artist_match_count_raw_release": r["artist_match_count_raw"],
            "music_text_list_baseline": b["artist_text_list"],
            "music_text_joined_baseline": b["artist_text_joined"],
            "music_text_list_release": r["artist_text_list"],
            "music_text_joined_release": r["artist_text_joined"],
        })

    return pd.DataFrame(out_rows)



# MAIN

if __name__ == "__main__":
    print("\n[1] Reading base table and preparing song fields...")
    base_df = pd.read_csv(BASE_TABLE_CSV)
    song_df = prepare_song_table_from_base(BASE_TABLE_CSV)

    prep_preview_path = os.path.join(OUT_DIR, "songs_prepared_for_matching_preview.csv")
    song_df[
        [
            "song_id", "artist", "song", "album", "release_date",
            "baseline_start", "baseline_end", "release_start", "release_end",
            "artist_norm", "song_norm"
        ]
    ].to_csv(prep_preview_path, index=False)
    print(f"Saved preview: {prep_preview_path}")

    if WRITE_SHRUNK_FILES:
        print("\n[2] Shrinking music submissions...")
        shrink_reddit_zst(
            input_zst=MUSIC_ZST,
            output_zst=MUSIC_SHRUNK_ZST,
            min_date=MIN_POST_DATE,
            max_date=MAX_POST_DATE
        )

    music_match_file = (
        MUSIC_SHRUNK_ZST
        if (USE_SHRUNK_FOR_MATCHING and os.path.exists(MUSIC_SHRUNK_ZST))
        else MUSIC_ZST
    )

    print("\n[3] Matching music submissions...")
    music_agg = process_music_subreddit(
        zst_path=music_match_file,
        song_df=song_df,
    )
    reddit_features = agg_to_music_feature_df(song_df, music_agg)

    preferred_order = [
        "song_id",
        "music_volume_baseline", "music_volume_release",
        "music_controversy_baseline", "music_controversy_release",
        "music_artist_match_count_raw_baseline", "music_artist_match_count_raw_release",
        "music_text_joined_baseline", "music_text_joined_release",
        "music_text_list_baseline", "music_text_list_release",
    ]
    reddit_features = reddit_features[preferred_order]

    reddit_features.to_csv(REDDIT_FEATURES_ONLY_CSV, index=False)
    print(f"Saved: {REDDIT_FEATURES_ONLY_CSV}")

    print("\n[4] Enriching base table without modifying the original file...")
    enriched_df = base_df.merge(reddit_features, on="song_id", how="left", validate="m:1")
    enriched_df.to_csv(ENRICHED_OUTPUT_CSV, index=False)
    print(f"Saved: {ENRICHED_OUTPUT_CSV}")

    run_summary = {
        "base_table_input": BASE_TABLE_CSV,
        "base_table_rows": int(len(base_df)),
        "matching_song_rows": int(len(song_df)),
        "music_input": MUSIC_ZST,
        "music_shrunk_input": MUSIC_SHRUNK_ZST,
        "reddit_features_only_output": REDDIT_FEATURES_ONLY_CSV,
        "enriched_output_without_overwriting_original": ENRICHED_OUTPUT_CSV,
        "write_shrunk_files": WRITE_SHRUNK_FILES,
        "max_texts_per_bucket": MAX_TEXTS_PER_BUCKET,
        "matching_logic": "artist_only"
    }

    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {summary_path}")
    print("\nDone.")
