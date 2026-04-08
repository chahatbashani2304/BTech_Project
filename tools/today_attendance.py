import argparse
import csv
import sqlite3
from datetime import date
from pathlib import Path


def fetch_attendance(conn: sqlite3.Connection, date_str: str):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.name, a.timestamp, a.confidence
        FROM attendance_logs a
        JOIN students s ON s.id = a.student_id
        WHERE date(a.timestamp) = ?
        ORDER BY a.timestamp ASC
        """,
        (date_str,),
    )
    return cur.fetchall()


def print_table(rows, date_str: str) -> None:
    print(f"Attendance logs for {date_str}")
    print("-" * 74)
    print(f"{'Person_name':25} {'timestamp':22} {'confidence':12}")
    print("-" * 74)
    for employee, timestamp, confidence in rows:
        print(f"{employee[:25]:25} {timestamp:22} {confidence:<12.6f}")
    print("-" * 74)
    print(f"Total rows: {len(rows)}")


def write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["employee", "timestamp", "confidence"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show attendance for a specific date from SQLite database.")
    parser.add_argument("--db-path", type=Path, default=Path("attendance.db"), help="Path to SQLite database.")
    parser.add_argument("--date", dest="date_str", default=date.today().isoformat(), help="Date in YYYY-MM-DD.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    db_path = args.db_path.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    rows = fetch_attendance(conn, args.date_str)
    conn.close()

    if not rows:
        print(f"No attendance rows found for {args.date_str}.")
        return

    print_table(rows, args.date_str)
    if args.csv_out:
        write_csv(rows, args.csv_out.resolve())
        print(f"CSV exported to: {args.csv_out.resolve()}")


if __name__ == "__main__":
    main()
