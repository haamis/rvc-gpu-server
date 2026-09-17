#!/usr/bin/env python3
"""Search voice-models.com for RVC voice models, download them, and add
them to config/voices.yaml.

Usage:
  python tools/rvc_models.py search <query> [--page N]
  python tools/rvc_models.py add <query|model-url|zip-url> [--as NAME] [--pick N] [--force]
  python tools/rvc_models.py add-zip <local.zip> [--as NAME] [--force]

`add` resolves the argument to a model zip (search terms -> first/picked
result; voice-models.com model page -> its download link; anything else is
treated as a direct zip URL), downloads and unpacks the .pth/.index into
rvc_models/<name>/, then prints the voices.yaml block for the THIN client's
config/voices.yaml (paste it there; this tool never edits the thin config
itself — use --config only when the thin checkout is mounted locally).
Google Drive folder links are not supported — download
manually and use `add-zip`. Restart the bot afterwards to pick up the new
voice.
"""
import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://voice-models.com"
UA = {"User-Agent": "Mozilla/5.0 (ttsbot-rvc-models)"}
DEFAULT_MODEL_DIR = PROJECT_ROOT / "rvc_models"
DIRECT_HOSTS = (r"huggingface\.co", r"cdn\.discordapp\.com", r"www\.mediafire\.com")
VOICE_KEY_RE = re.compile(r"[a-z][a-z0-9_-]*")

DEFAULT_ENTRY = {
    "pitch": 0,
    "index_rate": 0.75,
    "f0_method": "pm",
    "speaker_id": 0,
    "speed_cloud": 1.0,
    "speed_local": 0.7,
}


class ModelRow:
    def __init__(self, model_id: str, title: str, size: str, zip_url: str):
        self.model_id = model_id
        self.title = title
        self.size = size
        self.zip_url = zip_url

    def __str__(self):
        host = re.match(r"https?://([^/]+)", self.zip_url)
        host = host.group(1) if host else "?"
        return f"{self.title}  ({self.size or '?'}; {host}) [{self.model_id}]"


def _get(url: str, **kw) -> requests.Response:
    resp = requests.get(url, headers=UA, timeout=60, **kw)
    resp.raise_for_status()
    return resp


# --- voice-models.com -------------------------------------------------------


def search_models(query: str, page: int = 1) -> list[ModelRow]:
    """POST the site's own search endpoint and parse the result table."""
    resp = requests.post(
        f"{BASE_URL}/fetch_data.php",
        data={"page": page, "search": query},
        headers=UA,
        timeout=60,
    )
    resp.raise_for_status()
    table = resp.json().get("table", "")
    rows: list[ModelRow] = []
    for chunk in table.split("<tr>")[1:]:
        id_m = re.search(r"href='(/model/[A-Za-z0-9]+)'", chunk)
        if not id_m:
            continue
        title_blob = re.search(r"href='/model/[A-Za-z0-9]+'[^>]*>(.*?)</a>", chunk, re.S)
        title = " ".join(re.sub(r"<[^>]+>", " ", title_blob.group(1)).split()) if title_blob else ""
        size_m = re.search(r"badge bg-secondary[^>]*>([^<]+)<", chunk)
        url_m = re.search(r"data-clipboard-text='([^']+)'", chunk)
        rows.append(
            ModelRow(
                model_id=id_m.group(1).rsplit("/", 1)[-1],
                title=title,
                size=size_m.group(1).strip() if size_m else "",
                zip_url=url_m.group(1) if url_m else "",
            )
        )
    return rows


def resolve_download_from_model_page(model_id: str) -> str:
    """Scrape the /model/<id> page for its direct download link."""
    html = _get(f"{BASE_URL}/model/{model_id}").text
    m = re.search(rf"href=\"(https?://(?:{'|'.join(DIRECT_HOSTS)})[^\"]+)\"", html)
    if not m:
        raise SystemExit(
            f"No direct download link found on /model/{model_id} "
            "(Google Drive models must be downloaded manually; use add-zip)."
        )
    return m.group(1).replace("&amp;", "&")


def resolve_zip_url(arg: str, pick: int | None) -> tuple[str, str]:
    """Resolve a search query / model URL / direct URL to (zip_url, label)."""
    if re.match(rf"https?://({'|'.join(DIRECT_HOSTS)})/", arg):
        return arg, arg.rsplit("/", 1)[-1]
    if re.match(rf"{BASE_URL}/model/", arg):
        return resolve_download_from_model_page(arg.rstrip("/").rsplit("/", 1)[-1]), arg
    rows = search_models(arg)
    if not rows:
        raise SystemExit(f"No results for {arg!r} on voice-models.com")
    if len(rows) == 1 or pick is not None:
        row = rows[(pick or 1) - 1]
    else:
        print("Multiple results — re-run with --pick N:")
        for i, row in enumerate(rows, 1):
            print(f"  {i}) {row}")
        raise SystemExit(2)
    return row.zip_url, row.title


# --- downloading / unpacking -------------------------------------------------


def download_zip(url: str, dest: Path) -> Path:
    if "drive.google.com" in url:
        file_id = re.search(r"/file/d/([^/]+)|[?&]id=([^&]+)", url)
        if not file_id or "folders" in url:
            raise SystemExit(
                "Google Drive folder links are not supported — download the "
                "zip manually and use `add-zip <file>`."
            )
        gid = file_id.group(1) or file_id.group(2)
        return _download_gdrive(gid, dest)
    return _stream_to_file(url, dest)


def _stream_to_file(url: str, dest: Path, session: requests.Session | None = None) -> Path:
    getter = session or requests
    with getter.get(url, headers=UA, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", ""):
            raise SystemExit(f"Got an HTML page instead of a zip from {url}")
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def _download_gdrive(file_id: str, dest: Path) -> Path:
    """Google Drive single-file download incl. the virus-scan confirm page."""
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, headers=UA, stream=True, timeout=120)
    if "text/html" in resp.headers.get("content-type", ""):
        form = re.search(r"<form[^>]+action=\"([^\"]+)\"", resp.text)
        if not form:
            raise SystemExit("Google Drive download page has no confirm form (file private?)")
        fields = dict(re.findall(r"name=\"([^\"]+)\" value=\"([^\"]*)\"", resp.text))
        url = form.group(1).replace("&amp;", "&")
        resp = session.get(url, params=fields, headers=UA, stream=True, timeout=300)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    head = open(dest, "rb").read(4)
    if head[:2] != b"PK":
        raise SystemExit("Google Drive did not yield a zip (file too large for the confirm flow, or private). Use add-zip.")
    return dest


def pick_model_files(zip_path: Path) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo | None]:
    """Choose the voice .pth and (optionally) the .index inside an RVC zip.

    Training checkpoints (G_*/D_*, files under logs/) are skipped; when
    several candidates remain the largest is taken.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [i for i in zf.infolist() if not i.is_dir()]

    def candidate(info: zipfile.ZipInfo) -> bool:
        base = Path(info.filename).name
        return not (base.startswith(("G_", "D_")) or "logs" in Path(info.filename).parts)

    pths = sorted(
        (i for i in names if i.filename.endswith(".pth") and candidate(i)),
        key=lambda i: i.file_size,
        reverse=True,
    )
    if not pths:
        raise SystemExit(f"No usable .pth model found in {zip_path.name} "
                         "(maybe only training checkpoints — inspect the zip manually).")
    indexes = sorted(
        (i for i in names if i.filename.endswith(".index") and candidate(i)),
        key=lambda i: i.file_size,
        reverse=True,
    )
    return pths[0], indexes[0] if indexes else None


def install_model(zip_path: Path, voice_dir: Path, force: bool) -> tuple[Path, Path | None]:
    """Extract the chosen .pth/.index into voice_dir (one dir per voice)."""
    pth, index = pick_model_files(zip_path)
    voice_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for info in (pth, index):
        if info is None:
            continue
        target = voice_dir / Path(info.filename).name
        if target.exists() and not force:
            raise SystemExit(f"{target} already exists (use --force to overwrite)")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        targets.append(target)
    return targets[0], targets[1] if len(targets) > 1 else None


# --- config update (comment-preserving) --------------------------------------


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")
    return slug or "voice"


GENERIC_ZIP_NAMES = {"model", "voice", "voices", "rvc", "weights", "checkpoint"}


def default_voice_name(title: str, zip_url: str) -> str:
    stem = Path(zip_url).stem or ""
    slug = slugify(stem)
    if not slug or slug in GENERIC_ZIP_NAMES:
        slug = slugify(title)
    return slug[:40] or "voice"


def format_voice_entry(name: str, pth: Path, index: Path | None, provenance: str) -> str:
    lines = [
        f"  {name}:",
        f"    tts: en_US-lessac-medium",
        f"    rvc_model: {pth}",
    ]
    if index is not None:
        lines.append(f"    rvc_index: {index}")
    for key, value in DEFAULT_ENTRY.items():
        lines.append(f"    {key}: {value}")
    lines.append(f"    # Added by tools/rvc_models.py — {provenance}")
    return "\n".join(lines) + "\n"


def insert_voice_block(config_path: Path, name: str, block: str) -> None:
    """Append the voice inside the `voices:` section, preserving comments.

    The block is inserted after the last entry of the `voices:` mapping (or
    before the next top-level key if one follows). The result is validated
    with yaml.safe_load before the file is written.
    """
    text = config_path.read_text()
    lines = text.splitlines(keepends=True)
    voices_at = next((i for i, l in enumerate(lines) if re.match(r"^voices:\s*(#.*)?$", l)), None)
    if voices_at is None:
        raise SystemExit(f"No 'voices:' section found in {config_path}")

    insert_at = len(lines)
    for j in range(voices_at + 1, len(lines)):
        if lines[j].strip() and not lines[j][0].isspace():
            insert_at = j  # next top-level key
            break
    lines.insert(insert_at, block if block.endswith("\n") else block + "\n")

    candidate_text = "".join(lines)
    data = yaml.safe_load(candidate_text)
    if not isinstance(data, dict) or name not in (data.get("voices") or {}):
        raise SystemExit("Internal error: rewritten config does not contain the new voice; aborting.")
    backup = config_path.with_suffix(".yaml.bak")
    shutil.copy2(config_path, backup)
    config_path.write_text(candidate_text)
    print(f"Config updated (backup at {backup.name}).")


def add_voice_to_config(
    config_path: Path,
    name: str,
    pth: Path,
    index: Path | None,
    provenance: str,
    force: bool,
) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    voices = (cfg or {}).get("voices") or {}
    if name in voices and not force:
        raise SystemExit(f"Voice '{name}' already exists in {config_path} (use --force to replace)")
    if not VOICE_KEY_RE.fullmatch(name):
        raise SystemExit(f"Voice name '{name}' must match {VOICE_KEY_RE.pattern} (lowercase letters, digits, _ or -)")
    block = format_voice_entry(name, pth, index, provenance)
    if name in voices and force:
        raise SystemExit("--force replacement of an existing voice is not supported; remove it from the config first.")
    insert_voice_block(config_path, name, block)


# --- command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="search voice-models.com")
    p_search.add_argument("query")
    p_search.add_argument("--page", type=int, default=1)

    def add_common(p):
        p.add_argument("--as", dest="name", help="voice name (default: derived from the zip)")
        p.add_argument("--models-dir", type=Path, default=DEFAULT_MODEL_DIR)
        p.add_argument("--config", type=Path, default=None,
                       help="thin-client voices.yaml to update in place; when omitted, "
                            "the voices.yaml block is printed for manual pasting")
        p.add_argument("--force", action="store_true", help="overwrite existing model files")

    p_add = sub.add_parser("add", help="search/download a model and add it to the config")
    p_add.add_argument("target", help="search terms, voice-models.com model URL, or direct zip URL")
    p_add.add_argument("--pick", type=int, help="1-based result index when several match")
    add_common(p_add)

    p_zip = sub.add_parser("add-zip", help="add a manually downloaded model zip")
    p_zip.add_argument("zip", type=Path)
    add_common(p_zip)

    args = parser.parse_args(argv)

    if args.cmd == "search":
        for i, row in enumerate(search_models(args.query, args.page), 1):
            print(f"{i}) {row}")
        return

    provenance: str
    if args.cmd == "add":
        zip_url, label = resolve_zip_url(args.target, args.pick)
        print(f"Downloading: {zip_url}")
        with tempfile.TemporaryDirectory(prefix="rvc_models_") as tmp:
            zip_path = Path(tmp) / "model.zip"
            download_zip(zip_url, zip_path)
            print(f"Downloaded {zip_path.stat().st_size / 1e6:.1f} MB ({label})")
            name = args.name or default_voice_name(label, zip_url)
            pth, index = install_model(zip_path, args.models_dir / name, args.force)
            provenance = f"https://voice-models.com/model/ {label!r}"
    else:
        name = args.name or default_voice_name(args.zip.stem, str(args.zip))
        pth, index = install_model(args.zip, args.models_dir / name, args.force)
        provenance = f"local zip {args.zip.name!r}"

    print(f"Installed: {pth}")
    if index:
        print(f"Index:     {index}")
    else:
        print("No .index in the zip — the voice will run without an index blend.")
    if args.config is not None:
        add_voice_to_config(args.config, name, pth, index, provenance, args.force)
        print(f"Voice '{name}' added. Restart the bot to pick it up.")
    else:
        # Normal case: the thin client's voices.yaml lives on another
        # machine, so print the block for manual pasting instead.
        print("Add this to the thin client's config/voices.yaml under `voices:`:")
        print(format_voice_entry(name, pth, index, provenance))


if __name__ == "__main__":
    main()
