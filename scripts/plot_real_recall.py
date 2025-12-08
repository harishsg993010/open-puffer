#!/usr/bin/env python3
"""
Plot recall benchmark results for real embeddings.

Usage:
    python plot_real_recall.py recall_real.csv [--output-dir ./plots]

Outputs:
    - real_recall_vs_nprobe.png: Recall@K vs nprobe
    - real_latency_vs_nprobe.png: Latency percentiles vs nprobe
    - real_recall_latency_tradeoff.png: Recall vs latency tradeoff curve
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_recall_vs_nprobe(df: pd.DataFrame, output_path: Path) -> None:
    """Plot recall@K vs nprobe."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['nprobe'], df['mean_recall'], 'bo-', linewidth=2, markersize=8, label='Mean Recall')
    ax.fill_between(df['nprobe'], df['mean_recall'] * 0.98, df['mean_recall'].clip(upper=1.0), alpha=0.2)

    ax.set_xlabel('nprobe', fontsize=12)
    ax.set_ylabel(f'Recall@{df["top_k"].iloc[0]}', fontsize=12)
    ax.set_title('Real Embeddings: Recall vs nprobe', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Add value annotations
    for i, row in df.iterrows():
        ax.annotate(f'{row["mean_recall"]:.3f}',
                   (row['nprobe'], row['mean_recall']),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_vs_nprobe(df: pd.DataFrame, output_path: Path) -> None:
    """Plot latency percentiles vs nprobe."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['nprobe'], df['p50_ms'], 'g^-', linewidth=2, markersize=8, label='P50')
    ax.plot(df['nprobe'], df['p95_ms'], 'yo-', linewidth=2, markersize=8, label='P95')
    ax.plot(df['nprobe'], df['p99_ms'], 'rs-', linewidth=2, markersize=8, label='P99')
    ax.plot(df['nprobe'], df['mean_ms'], 'b+-', linewidth=2, markersize=8, label='Mean')

    ax.set_xlabel('nprobe', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Real Embeddings: Search Latency vs nprobe', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_recall_latency_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    """Plot recall vs latency tradeoff (Pareto frontier)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main scatter with line
    scatter = ax.scatter(df['p50_ms'], df['mean_recall'],
                        c=df['nprobe'], cmap='viridis', s=150, zorder=3)
    ax.plot(df['p50_ms'], df['mean_recall'], 'k--', alpha=0.5, zorder=2)

    # Annotate with nprobe values
    for i, row in df.iterrows():
        ax.annotate(f'nprobe={int(row["nprobe"])}',
                   (row['p50_ms'], row['mean_recall']),
                   textcoords="offset points", xytext=(10, 5),
                   ha='left', fontsize=9)

    ax.set_xlabel('P50 Latency (ms)', fontsize=12)
    ax.set_ylabel(f'Recall@{df["top_k"].iloc[0]}', fontsize=12)
    ax.set_title('Real Embeddings: Recall vs Latency Tradeoff', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('nprobe', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_qps_vs_nprobe(df: pd.DataFrame, output_path: Path) -> None:
    """Plot QPS vs nprobe."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(df['nprobe'].astype(str), df['qps'], color='steelblue', edgecolor='black')

    ax.set_xlabel('nprobe', fontsize=12)
    ax.set_ylabel('Queries Per Second (QPS)', fontsize=12)
    ax.set_title('Real Embeddings: Throughput vs nprobe', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value annotations
    for i, (x, y) in enumerate(zip(df['nprobe'].astype(str), df['qps'])):
        ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot real embeddings recall benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('csv_file', type=Path, help='Input CSV file from real-recall-bench')
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('.'),
                       help='Output directory for plots (default: current directory)')

    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f"Error: CSV file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} rows from {args.csv_file}")
    print(df.to_string(index=False))
    print()

    # Generate plots
    plot_recall_vs_nprobe(df, args.output_dir / 'real_recall_vs_nprobe.png')
    plot_latency_vs_nprobe(df, args.output_dir / 'real_latency_vs_nprobe.png')
    plot_recall_latency_tradeoff(df, args.output_dir / 'real_recall_latency_tradeoff.png')
    plot_qps_vs_nprobe(df, args.output_dir / 'real_qps_vs_nprobe.png')

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
