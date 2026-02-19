import argparse
import csv
import sqlite3
from datetime import date
from pathlib import Path


def fetch_attendance(conn: sqlite3.Connection, date_str: str):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.name, a.status, a.first_seen_at, a.last_seen_at
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.date = ?
        ORDER BY s.name ASC
        """,
        (date_str,),
    )
    return cur.fetchall()


def print_table(rows, date_str: str) -> None:
    print(f"Attendance for {date_str}")
    print("-" * 72)
    print(f"{'Name':30} {'Status':10} {'First Seen':15} {'Last Seen':15}")
    print("-" * 72)
    for name, status, first_seen, last_seen in rows:
        first_seen_short = (first_seen or "-")[-8:] if first_seen else "-"
        last_seen_short = (last_seen or "-")[-8:] if last_seen else "-"
        print(f"{name[:30]:30} {status:10} {first_seen_short:15} {last_seen_short:15}")
    print("-" * 72)
    present = sum(1 for _, status, _, _ in rows if status == "present")
    absent = sum(1 for _, status, _, _ in rows if status == "absent")
    print(f"Total: {len(rows)} | Present: {present} | Absent: {absent}")


def write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "status", "first_seen_at", "last_seen_at"])
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
