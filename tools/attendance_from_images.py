import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

from mtcnn import MTCNN


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            faces_detected INTEGER NOT NULL,
            scanned_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('present', 'absent')),
            first_seen_at TEXT,
            last_seen_at TEXT,
            UNIQUE(student_id, date),
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
        """
    )
    conn.commit()
    return conn


def get_or_create_student(cur: sqlite3.Cursor, name: str) -> int:
    cur.execute("INSERT OR IGNORE INTO students(name) VALUES (?)", (name,))
    cur.execute("SELECT id FROM students WHERE name = ?", (name,))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Could not create or fetch student: {name}")
    return int(row[0])


def upsert_attendance(cur: sqlite3.Cursor, student_id: int, date_str: str, seen_at: str, status: str) -> None:
    cur.execute(
        """
        INSERT INTO attendance(student_id, date, status, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(student_id, date) DO UPDATE SET
            status = CASE
                WHEN excluded.status = 'present' THEN 'present'
                ELSE attendance.status
            END,
            first_seen_at = COALESCE(attendance.first_seen_at, excluded.first_seen_at),
            last_seen_at = CASE
                WHEN excluded.status = 'present' THEN excluded.last_seen_at
                ELSE attendance.last_seen_at
            END
        """,
        (student_id, date_str, status, seen_at if status == "present" else None, seen_at if status == "present" else None),
    )


def iter_images(root: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        yield from root.rglob(ext)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan image dataset and update attendance in SQLite.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("tests/images/lpw_small"),
        help="Root folder containing person subfolders with images.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("attendance.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--device",
        default="CPU:0",
        help='MTCNN device, e.g. "CPU:0" or "CUDA:0".',
    )
    parser.add_argument(
        "--no-reset-today",
        action="store_true",
        help="Do not clear today's previous attendance/scans before processing.",
    )
    args = parser.parse_args()

    images_root = args.images_root.resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    detector = MTCNN(device=args.device)
    conn = init_db(args.db_path.resolve())
    cur = conn.cursor()

    today = datetime.now().date().isoformat()
    if not args.no_reset_today:
        cur.execute("DELETE FROM attendance WHERE date = ?", (today,))
        cur.execute("DELETE FROM scans WHERE date(scanned_at) = ?", (today,))
        conn.commit()

    total_images = 0
    present_updates = 0

    for image_path in iter_images(images_root):
        total_images += 1
        person_name = image_path.parent.name
        student_id = get_or_create_student(cur, person_name)

        detections = detector.detect_faces(str(image_path))
        faces_detected = len(detections)
        now_iso = datetime.now().isoformat(timespec="seconds")

        cur.execute(
            """
            INSERT INTO scans(student_id, image_path, faces_detected, scanned_at)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, str(image_path), faces_detected, now_iso),
        )

        status = "present" if faces_detected > 0 else "absent"
        upsert_attendance(cur, student_id, today, now_iso, status)
        if status == "present":
            present_updates += 1

    conn.commit()
    conn.close()

    print(f"Scanned images: {total_images}")
    print(f"Attendance rows updated as present: {present_updates}")
    print(f"Database: {args.db_path.resolve()}")


if __name__ == "__main__":
    main()
