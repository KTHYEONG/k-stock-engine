import argparse
import time
from pathlib import Path

# Logs root directory
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
SCRATCH_LOGS_DIR = LOGS_DIR / "scratch"

# Patterns of temporary/one-off log files to clean from logs root
TRANSIENT_PATTERNS = [
    "scratch_*.log",
    "exp_*.log",
    "prep_*.log",
    "prep_*.out",
    "verify_*.log",
    "spec_*.log",
    "composite_*.log",
    "run_full_*.log",
    "metrics_*_run*.out",
    "oi_*_run*.out",
    "smart_money_*.out",
]


def clean_logs(retention_days: int = 3, dry_run: bool = False) -> None:
    """Clean transient log files from logs/ root and logs/scratch/."""
    if not LOGS_DIR.exists():
        print("No logs directory found.")
        return

    now = time.time()
    cutoff_time = now - (retention_days * 86400)
    removed_count = 0
    reclaimed_bytes = 0

    # 1. Clean matching transient files in logs/ root
    for pattern in TRANSIENT_PATTERNS:
        for file_path in LOGS_DIR.glob(pattern):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                if dry_run:
                    print(f"[DRY-RUN] Would remove root transient: {file_path.name} ({file_size} bytes)")
                else:
                    file_path.unlink()
                    print(f"[REMOVED] Transient log: {file_path.name}")
                removed_count += 1
                reclaimed_bytes += file_size

    # 2. Clean old files inside logs/scratch/
    if SCRATCH_LOGS_DIR.exists():
        for file_path in SCRATCH_LOGS_DIR.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_size = file_path.stat().st_size
                if dry_run:
                    print(f"[DRY-RUN] Would remove scratch log: {file_path.name}")
                else:
                    file_path.unlink()
                    print(f"[REMOVED] Old scratch log: {file_path.name}")
                removed_count += 1
                reclaimed_bytes += file_size

    print(f"Cleanup complete. Removed {removed_count} files ({reclaimed_bytes / (1024*1024):.2f} MB reclaimed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean transient log files in logs/")
    parser.add_argument("--days", type=int, default=3, help="Retention period in days for scratch logs (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be removed without deleting them")
    args = parser.parse_args()

    clean_logs(retention_days=args.days, dry_run=args.dry_run)
