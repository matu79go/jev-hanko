"""デモページをヘッドレスブラウザで再生して録画し、mp4 と gif にする。

Usage: python3 scripts/record_demo.py race [en]   # docs/index.html → media/demo[_en].*      (速度の競走)
       python3 scripts/record_demo.py sort [en]   # docs/sort.html  → media/demo_sort[_en].* (カゴへの振り分け)
要 playwright + chromium、ffmpeg。ページ側は実測の待ち時間をそのまま再生する。
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"race": ("index.html", "demo", 1280, 900), "sort": ("sort.html", "demo_sort", 1280, 720)}


def main(which, lang="ja"):
    html, stem, w, h = TARGETS[which]
    stem += "_en" if lang == "en" else ""
    media = ROOT / "media"
    media.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=1,
                                  record_video_dir=str(tmp), record_video_size={"width": w, "height": h})
        page = ctx.new_page()
        page.goto((ROOT / "docs" / html).as_uri() + "?record=1" + ("&lang=en" if lang == "en" else ""))
        page.wait_for_function("window.__done === true", timeout=180_000)
        page.screenshot(path=str(media / f"{stem}_final.png"))
        ctx.close()
        browser.close()
    webm = next(tmp.glob("*.webm"))
    mp4, gif = media / f"{stem}.mp4", media / f"{stem}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
    # Qiita の画像上限(10MB)に収まるよう、長い方は少し粗くする
    fps, width, colors = (8, 900, 80) if which == "race" else (8, 800, 64)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-vf",
                    f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(gif)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    for f in (mp4, gif):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "race", sys.argv[2] if len(sys.argv) > 2 else "ja")
