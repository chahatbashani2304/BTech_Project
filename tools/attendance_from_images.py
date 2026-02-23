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
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
        """
    )
    cur.execute("PRAGMA table_info(attendance_logs)")
    columns = [row[1] for row in cur.fetchall()]
    if "camera_id" in columns:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_logs_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        cur.execute(
            """
            INSERT INTO attendance_logs_new(id, student_id, image_path, timestamp, confidence)
            SELECT id, student_id, image_path, timestamp, confidence
            FROM attendance_logs
            """
        )
        cur.execute("DROP TABLE attendance_logs")
        cur.execute("ALTER TABLE attendance_logs_new RENAME TO attendance_logs")
    conn.commit()
    return conn


def get_or_create_student(cur: sqlite3.Cursor, name: str) -> int:
    cur.execute("INSERT OR IGNORE INTO students(name) VALUES (?)", (name,))
    cur.execute("SELECT id FROM students WHERE name = ?", (name,))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Could not create or fetch student: {name}")
    return int(row[0])


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
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Only store rows with confidence >= this threshold.",
    )
    parser.add_argument(
        "--no-reset-today",
        action="store_true",
        help="Do not clear today's previous logs/scans before processing.",
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
        cur.execute("DELETE FROM attendance_logs WHERE date(timestamp) = ?", (today,))
        cur.execute("DELETE FROM scans WHERE date(scanned_at) = ?", (today,))
        conn.commit()

    total_images = 0
    logged_rows = 0

    for image_path in iter_images(images_root):
        total_images += 1
        person_name = image_path.parent.name
        student_id = get_or_create_student(cur, person_name)

        detections = detector.detect_faces(str(image_path))
        faces_detected = len(detections)
        now_iso = datetime.now().isoformat(timespec="seconds")
        max_confidence = max((float(d.get("confidence", 0.0)) for d in detections), default=0.0)

        cur.execute(
            """
            INSERT INTO scans(student_id, image_path, faces_detected, scanned_at)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, str(image_path), faces_detected, now_iso),
        )

        if faces_detected > 0 and max_confidence >= args.confidence_threshold:
            cur.execute(
                """
                INSERT INTO attendance_logs(student_id, image_path, timestamp, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, str(image_path), now_iso, max_confidence),
            )
            logged_rows += 1

    conn.commit()
    conn.close()

    print(f"Scanned images: {total_images}")
    print(f"Attendance rows logged: {logged_rows}")
    print(f"Database: {args.db_path.resolve()}")


if __name__ == "__main__":
    main()
