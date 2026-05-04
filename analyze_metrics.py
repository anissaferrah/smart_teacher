"""
Smart Teacher — Performance Metrics Analysis Tool.

Analyzes CSV logs from learning sessions and generates comprehensive
visualization dashboard for KPI tracking and performance optimization.

Metrics Tracked:
    - Total response time distribution (vs KPI 5s threshold)
    - Component timing breakdown (STT / LLM / TTS)
    - Performance by language
    - KPI compliance rate
    - Real-Time Factor (RTF) for STT
    - Temporal trends

Output:
    - Console statistics and summaries
    - Multi-panel matplotlib dashboard
    - PNG export to logs/metrics_dashboard.png

Usage:
    # Analyze main metrics.csv
    python analyze_metrics.py
    
    # Include STT-specific analysis
    python analyze_metrics.py --stt
    
    # Analyze custom CSV
    python analyze_metrics.py <path_to_csv>

Requires:
    - pandas, matplotlib, numpy
    - CSV files from logger (metrics.csv, stt_metrics.csv)

Output Files:
    - Dashboard figure displayed/saved
    - Statistics printed to stdout
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from core.config import Config

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SmartTeacher.Metrics")

METRICS_CSV: str = Config.CSV_LOG_FILE
STT_CSV: str = Config.STT_LOG_FILE
KPI_LIMIT: float = Config.MAX_RESPONSE_TIME  # 5.0s
RTF_LIMIT: float = Config.TARGET_RTF  # 0.5

PALETTE: dict = {
    "stt": "#4e79a7",
    "llm": "#f28e2b",
    "tts": "#59a14f",
    "total": "#e15759",
    "kpi": "#76b7b2",
}


# ════════════════════════════════════════════════════════════════════════
# GLOBAL METRICS ANALYSIS
# ════════════════════════════════════════════════════════════════════════


def analyze_global(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Analyze global performance metrics from main CSV.
    
    Generates console statistics and visualization dashboard.
    
    Parameters
    ----------
    csv_path : str
        Path to metrics.csv file
    
    Returns
    -------
    pd.DataFrame or None
        Loaded dataframe if successful, None if file not found/empty
    """
    if not os.path.exists(csv_path):
        log.error(f"File not found: {csv_path}")
        log.info("Launch server and perform interactions first.")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        log.warning("CSV is empty — no data to analyze")
        return None

    # Console statistics
    print("\n" + "=" * 70)
    print("📊 SMART TEACHER — PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(f"\n📁 File: {csv_path}")
    print(f"📈 Interactions: {len(df)}")

    # Timing statistics
    timing_cols = [c for c in ["stt_time", "llm_time", "tts_time", "total_time"] if c in df.columns]
    if timing_cols:
        print("\n⏱️  TIMING (seconds):")
        print(df[timing_cols].describe().round(3).to_string())

    # KPI compliance
    if "meets_kpi" in df.columns:
        kpi_rate = df["meets_kpi"].mean() * 100
        print(f"\n🎯 KPI Compliance (<{KPI_LIMIT}s): {kpi_rate:.1f}%")

    # Language distribution
    if "language" in df.columns:
        print("\n🌍 Languages Detected:")
        print(df["language"].value_counts().to_string())

    # Visualization
    create_dashboard(df)

    return df


def create_dashboard(df: pd.DataFrame) -> None:
    """
    Create multi-panel performance dashboard.
    
    Parameters
    ----------
    df : pd.DataFrame
        Metrics dataframe
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("SmartTeacher — Performance Dashboard", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Total time distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if "total_time" in df.columns:
        ax1.hist(
            df["total_time"].dropna(),
            bins=20,
            color=PALETTE["total"],
            edgecolor="white",
            alpha=0.85
        )
        ax1.axvline(
            KPI_LIMIT,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"KPI = {KPI_LIMIT}s"
        )
        ax1.set_title("Total Response Time Distribution")
        ax1.set_xlabel("Seconds")
        ax1.set_ylabel("Count")
        ax1.legend()

    # Panel 2: Component timing comparison
    ax2 = fig.add_subplot(gs[0, 1])
    comp_cols = [c for c in ["stt_time", "llm_time", "tts_time"] if c in df.columns]
    if comp_cols:
        means = df[comp_cols].mean()
        labels = [c.replace("_time", "").upper() for c in comp_cols]
        colors = [PALETTE.get(c.replace("_time", ""), "#999") for c in comp_cols]
        ax2.bar(labels, means, color=colors, edgecolor="white", alpha=0.85)
        ax2.set_title("Average Component Timing")
        ax2.set_ylabel("Seconds")
        ax2.tick_params(axis="x", rotation=45)

    # Panel 3: KPI compliance bar
    ax3 = fig.add_subplot(gs[0, 2])
    if "meets_kpi" in df.columns:
        kpi_counts = df["meets_kpi"].value_counts()
        labels = ["Meets KPI", "Exceeds KPI"]
        values = [kpi_counts.get(1, 0), kpi_counts.get(0, 0)]
        colors_kpi = [PALETTE["kpi"], "#d62728"]
        ax3.bar(labels, values, color=colors_kpi, edgecolor="white", alpha=0.85)
        ax3.set_title("KPI Compliance Distribution")
        ax3.set_ylabel("Count")

    # Panel 4: Performance by language
    ax4 = fig.add_subplot(gs[1, 0])
    if "language" in df.columns and "total_time" in df.columns:
        lang_perf = df.groupby("language")["total_time"].agg(["mean", "std"])
        ax4.bar(lang_perf.index, lang_perf["mean"], color=PALETTE["total"], alpha=0.85)
        ax4.set_title("Average Timing by Language")
        ax4.set_ylabel("Seconds")
        ax4.set_xlabel("Language")

    # Panel 5: Cumulative timing stacked
    ax5 = fig.add_subplot(gs[1, 1])
    time_cols = ["stt_time", "llm_time", "tts_time"]
    available_cols = [c for c in time_cols if c in df.columns]
    if available_cols:
        means = [df[c].mean() for c in available_cols]
        labels = [c.replace("_time", "").upper() for c in available_cols]
        colors = [PALETTE.get(c.replace("_time", ""), "#999") for c in available_cols]
        ax5.barh(["Average"], [sum(means)], color=colors[0], alpha=0.85, label=labels[0])
        for i, (mean, label, color) in enumerate(zip(means[1:], labels[1:], colors[1:])):
            ax5.barh(["Average"], [mean], left=sum(means[:i+1]), color=color, alpha=0.85, label=label)
        ax5.set_title("Total Time Composition")
        ax5.set_xlabel("Seconds")
        ax5.legend(loc="upper right")
        ax5.set_xlim([0, sum(means) * 1.1])

    # Panel 6: Time trend
    ax6 = fig.add_subplot(gs[1, 2])
    if "total_time" in df.columns:
        times = df["total_time"].dropna()
        if len(times) > 1:
            x = range(len(times))
            ax6.plot(x, times, marker="o", linestyle="-", color=PALETTE["total"], alpha=0.7)
            ax6.axhline(KPI_LIMIT, color="red", linestyle="--", linewidth=2, label=f"KPI={KPI_LIMIT}s")
            ax6.set_title("Response Time Over Time")
            ax6.set_xlabel("Interaction #")
            ax6.set_ylabel("Seconds")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

    # Save and display
    output_path = Path(Config.LOG_DIR) / "metrics_dashboard.png"
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    log.info(f"Dashboard saved: {output_path}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════
# STT-SPECIFIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════


def analyze_stt(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Analyze Speech-to-Text specific metrics (RTF, latency, accuracy).
    
    Parameters
    ----------
    csv_path : str
        Path to stt_metrics.csv file
    
    Returns
    -------
    pd.DataFrame or None
        Loaded dataframe if successful, None if file not found
    """
    if not os.path.exists(csv_path):
        log.warning(f"STT metrics file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        log.warning("STT CSV is empty")
        return None

    print("\n" + "=" * 70)
    print("🎤 STT METRICS ANALYSIS")
    print("=" * 70)
    print(f"Records: {len(df)}")

    if "rtf" in df.columns:
        avg_rtf = df["rtf"].mean()
        print("\n📊 Real-Time Factor (RTF):")
        print(f"   Average: {avg_rtf:.3f} ({avg_rtf < RTF_LIMIT and '✅' or '⚠️'} target={RTF_LIMIT})")
        print(f"   Range:   {df['rtf'].min():.3f} to {df['rtf'].max():.3f}")

    if "duration" in df.columns:
        print("\n⏱️  Audio Duration:")
        print(f"   Mean:    {df['duration'].mean():.2f}s")
        print(f"   Total:   {df['duration'].sum():.0f}s")

    if "confidence" in df.columns:
        print("\n📈 Confidence:")
        print(f"   Mean: {df['confidence'].mean():.3f}")
        print(f"   Min:  {df['confidence'].min():.3f}")

    print("=" * 70)
    return df


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════


def main() -> int:
    """
    Main analysis workflow.
    
    Returns
    -------
    int
        Exit code (0=success, 1=error)
    """
    parser = argparse.ArgumentParser(
        description="Smart Teacher — Analyze performance metrics"
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="Also analyze STT-specific metrics"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=METRICS_CSV,
        help="Path to metrics CSV file"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("📊 SMART TEACHER — METRICS ANALYZER")
    print("=" * 70)

    # Analyze main metrics
    df = analyze_global(args.csv)
    if df is None:
        return 1

    # Analyze STT if requested
    if args.stt:
        analyze_stt(STT_CSV)

    return 0


if __name__ == "__main__":
    sys.exit(main())
