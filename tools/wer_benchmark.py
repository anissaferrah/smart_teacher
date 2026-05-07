"""
Benchmark offline du WER STT sur un corpus de test étiqueté.

Usage :
    python tools/wer_benchmark.py [--corpus PATH] [--report PATH]

Le corpus est un dossier avec :
    corpus/
      manifest.json         # liste des items (chemin audio + transcription correcte)
      audio/0001.wav
      audio/0002.wav
      ...

Format manifest.json :
    [
      {"audio": "audio/0001.wav", "reference": "Bonjour, qu'est-ce que la régression linéaire ?", "language": "fr"},
      {"audio": "audio/0002.wav", "reference": "What is overfitting in machine learning?", "language": "en"},
      ...
    ]

Sortie : fichier JSON avec WER moyen, p50/p95, par langue, et erreurs détaillées.
Le seuil de validation est WER ≤ 5% (cahier des charges).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def load_manifest(corpus_dir: Path) -> list[dict]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERR] manifest.json missing in {corpus_dir}", file=sys.stderr)
        sys.exit(2)
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def transcribe_one(audio_path: Path, language: str | None, transcriber) -> tuple[str, float, float]:
    """Transcribe one audio file. Returns (text, stt_time, audio_duration)."""
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio_duration = len(audio) / sr

    t0 = time.time()
    text, stt_time, lang, lang_prob, _ = transcriber.transcribe(audio, language)
    return text or "", stt_time, audio_duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="dataset/wer_test_set",
                    help="Dossier corpus (avec manifest.json + audio/)")
    ap.add_argument("--report", default="logs/wer_benchmark_report.json",
                    help="Chemin du rapport JSON de sortie")
    ap.add_argument("--max-items", type=int, default=0,
                    help="Limite le nb d'items (0 = tous)")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"[ERR] corpus dir missing: {corpus_dir}", file=sys.stderr)
        print(f"      Crée-le avec un manifest.json + un dossier audio/", file=sys.stderr)
        sys.exit(2)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audio.transcriber import Transcriber
    from observability.wer import compute_wer

    print(f"📂 Loading manifest from {corpus_dir}…")
    items = load_manifest(corpus_dir)
    if args.max_items:
        items = items[: args.max_items]
    print(f"   {len(items)} items to transcribe")

    print("🎙️ Loading Whisper transcriber…")
    transcriber = Transcriber()

    results: list[dict] = []
    failures: list[dict] = []

    for i, item in enumerate(items, 1):
        audio_path = corpus_dir / item["audio"]
        if not audio_path.exists():
            print(f"   [{i:3}/{len(items)}] ⚠️ missing audio: {audio_path}")
            continue
        ref      = item.get("reference", "")
        language = item.get("language")

        try:
            hyp, stt_time, audio_dur = transcribe_one(audio_path, language, transcriber)
        except Exception as exc:
            print(f"   [{i:3}/{len(items)}] ❌ transcription failed: {exc}")
            failures.append({"audio": item["audio"], "error": str(exc)})
            continue

        wer_result = compute_wer(reference=ref, hypothesis=hyp)
        results.append({
            "audio":          item["audio"],
            "language":       language,
            "reference":      ref,
            "hypothesis":     hyp,
            "wer":            wer_result.wer,
            "cer":            wer_result.cer,
            "stt_time":       stt_time,
            "audio_duration": audio_dur,
            "rtf":            stt_time / audio_dur if audio_dur > 0 else 0,
            "meets_kpi":      wer_result.wer <= 0.05,
        })

        marker = "✅" if wer_result.wer <= 0.05 else "❌"
        print(f"   [{i:3}/{len(items)}] {marker} WER={wer_result.wer:.3f} ({language}) {audio_path.name}")

    # ── Aggregate ──────────────────────────────────────────────────────
    if not results:
        print("\n[WARN] no successful transcriptions — nothing to aggregate")
        sys.exit(1)

    import statistics

    def stats(vals: list[float]) -> dict:
        s = sorted(vals)
        return {
            "count": len(vals),
            "mean":  round(statistics.mean(vals), 4),
            "p50":   round(s[len(s)//2], 4),
            "p95":   round(s[int(len(s)*0.95)] if len(s) > 1 else s[0], 4),
            "max":   round(max(vals), 4),
        }

    wers = [r["wer"] for r in results]
    rtfs = [r["rtf"] for r in results if r["rtf"] > 0]

    by_lang: dict[str, list[float]] = {}
    for r in results:
        by_lang.setdefault(r["language"] or "unknown", []).append(r["wer"])

    total = len(results)
    passing = sum(1 for r in results if r["meets_kpi"])
    compliance = passing / total if total > 0 else 0.0

    report = {
        "summary": {
            "total_items":     total,
            "passing":         passing,
            "compliance_rate": round(compliance, 3),
            "kpi_target":      0.05,
            "global_wer":      stats(wers),
            "global_rtf":      stats(rtfs) if rtfs else None,
            "by_language":     {lang: stats(v) for lang, v in by_lang.items()},
            "failures":        len(failures),
        },
        "details":  results,
        "failures": failures,
    }

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total items     : {total}")
    print(f"  Passing (≤5% WER): {passing}/{total} ({compliance*100:.1f}%)")
    print(f"  Mean WER        : {report['summary']['global_wer']['mean']:.3f}")
    print(f"  p50 / p95 / max : "
          f"{report['summary']['global_wer']['p50']:.3f} / "
          f"{report['summary']['global_wer']['p95']:.3f} / "
          f"{report['summary']['global_wer']['max']:.3f}")
    if rtfs:
        print(f"  Mean RTF        : {report['summary']['global_rtf']['mean']:.3f}x")
    print(f"  By language     :")
    for lang, lst in by_lang.items():
        print(f"    {lang:8} → mean WER {sum(lst)/len(lst):.3f} ({len(lst)} items)")
    print()
    print(f"📄 Report saved to {out_path}")
    print("=" * 60)

    if compliance < 0.95:
        print(f"\n⚠️ Compliance rate {compliance*100:.1f}% < 95% — KPI WER non tenu sur ce corpus")
        sys.exit(1)
    else:
        print(f"\n✅ KPI WER < 5% tenu ({compliance*100:.1f}% des items)")


if __name__ == "__main__":
    main()
