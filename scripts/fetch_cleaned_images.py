#!/usr/bin/env python3
"""
Fetch poster images for the strict cleaned CSV.

This script is the single teammate bootstrap command after `git pull`.
By default it auto-discovers the project root and uses:
  - cleaned/MovieGenre_clean_with_images_full.csv
  - cleaned/downloaded_posters/

Common usage:
  python scripts/fetch_cleaned_images.py
  python scripts/fetch_cleaned_images.py --smoke-test
  python scripts/fetch_cleaned_images.py --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, *script_dir.parents, here, *here.parents]
    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        if (base / "MovieGenre.csv").is_file() and (base / "README.md").is_file():
            return base
    raise RuntimeError("Could not find project root containing MovieGenre.csv and README.md.")


def download_poster(url: str, out_path: Path, timeout: float, retries: int, sleep_s: float) -> tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached"

    if not isinstance(url, str) or not url.strip():
        return False, "empty_url"

    req = Request(url.strip(), headers={"User-Agent": "Mozilla/5.0"})
    last_error = "unknown"

    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout) as response:
                content = response.read()

            if not content:
                return False, "empty_body"

            out_path.write_bytes(content)
            return True, "downloaded"
        except HTTPError as exc:
            last_error = f"http_{exc.code}"
            # Most common terminal failures.
            if exc.code in {403, 404, 410}:
                return False, last_error
        except URLError as exc:
            last_error = f"urlerror_{exc.reason}"
        except Exception as exc:  # pragma: no cover - defensive
            last_error = f"other_{type(exc).__name__}"

        if attempt < retries and sleep_s > 0:
            sleep(sleep_s)

    return False, last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch posters for cleaned CSV rows.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to cleaned CSV containing imdbId and Poster columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory where downloaded posters are written.",
    )
    parser.add_argument("--timeout", type=float, default=6.0, help="HTTP timeout seconds per request.")
    parser.add_argument("--retries", type=int, default=0, help="Retries per URL after the first attempt.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between retries.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N rows.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke run: fetch first N rows and exit.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=10,
        help="Rows to process when --smoke-test is used.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify cached image coverage against the CSV (no downloads).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()
    csv_path = (args.csv_path if args.csv_path is not None else root / "cleaned/MovieGenre_clean_with_images_full.csv")
    out_dir = (args.out_dir if args.out_dir is not None else root / "cleaned/downloaded_posters")
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="latin-1")
    required_cols = {"imdbId", "Poster"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.verify:
        imdb_ids = pd.to_numeric(df["imdbId"], errors="coerce").dropna().astype(int).tolist()
        present = sum((out_dir / f"{imdb_id}.jpg").is_file() and (out_dir / f"{imdb_id}.jpg").stat().st_size > 0 for imdb_id in imdb_ids)
        print("Verify mode")
        print(f"CSV path: {csv_path}")
        print(f"Output dir: {out_dir}")
        print(f"Rows in CSV: {len(df)}")
        print(f"Valid imdbId rows: {len(imdb_ids)}")
        print(f"Cached non-empty jpg files matching imdbId: {present}")
        return

    if args.smoke_test:
        args.max_rows = args.smoke_rows
        print(f"Smoke test enabled: processing first {args.max_rows} rows")

    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()

    ok = 0
    fail = 0
    reasons: dict[str, int] = {}

    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            imdb_key = int(getattr(row, "imdbId"))
        except Exception:
            fail += 1
            reasons["bad_imdbId"] = reasons.get("bad_imdbId", 0) + 1
            continue

        out_path = out_dir / f"{imdb_key}.jpg"
        success, reason = download_poster(
            url=getattr(row, "Poster"),
            out_path=out_path,
            timeout=args.timeout,
            retries=args.retries,
            sleep_s=args.sleep,
        )
        if success:
            ok += 1
        else:
            fail += 1
            reasons[reason] = reasons.get(reason, 0) + 1

        if args.progress_every and i % args.progress_every == 0:
            print(f"Processed {i}/{len(df)} | ok={ok} fail={fail}")

    print("\nFetch complete")
    print(f"Rows processed: {len(df)}")
    print(f"Success (downloaded or cached): {ok}")
    print(f"Failed: {fail}")
    if reasons:
        top = sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:10]
        print(f"Top failure reasons: {top}")
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

