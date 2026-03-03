import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def dedupe_and_enforce_daily_uniqueness(cur: sqlite3.Cursor) -> None:
    # Keep only the latest row per (student_id, day), then enforce uniqueness.
    cur.execute(
        """
        DELETE FROM attendance_logs
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM attendance_logs
            GROUP BY student_id, date(timestamp)
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_student_day
        ON attendance_logs(student_id, date(timestamp))
        """
    )


def upsert_daily_attendance(
    cur: sqlite3.Cursor,
    student_id: int,
    image_path: str,
    now_iso: str,
    confidence: float,
) -> None:
    cur.execute(
        """
        SELECT id
        FROM attendance_logs
        WHERE student_id = ? AND date(timestamp) = date(?)
        ORDER BY id DESC
        LIMIT 1
        """,
        (student_id, now_iso),
    )
    row = cur.fetchone()
    if row is None:
        cur.execute(
            """
            INSERT INTO attendance_logs(student_id, image_path, timestamp, confidence)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, image_path, now_iso, confidence),
        )
    else:
        cur.execute(
            """
            UPDATE attendance_logs
            SET image_path = ?, timestamp = ?, confidence = ?
            WHERE id = ?
            """,
            (image_path, now_iso, confidence, int(row[0])),
        )


def iter_images(root: Path):
    allowed = {".jpg", ".jpeg", ".png"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def create_mtcnn_detector(device: str) -> MTCNN:
    try:
        return MTCNN(device=device)
    except TypeError:
        # Backward compatibility for older mtcnn versions that do not accept `device`.
        return MTCNN()


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

    conn = init_db(args.db_path.resolve())
    cur = conn.cursor()
    dedupe_and_enforce_daily_uniqueness(cur)
    conn.commit()
    detector = create_mtcnn_detector(args.device)

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

        try:
            detections = detector.detect_faces(str(image_path))
        except Exception as exc:
            print(f"Warning: could not read image {image_path} ({exc}); recording as 0 faces.")
            detections = []
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
            upsert_daily_attendance(cur, student_id, str(image_path), now_iso, max_confidence)
            logged_rows += 1

    conn.commit()
    conn.close()

    print(f"Scanned images: {total_images}")
    print(f"Attendance rows logged: {logged_rows}")
    print(f"Database: {args.db_path.resolve()}")


if __name__ == "__main__":
    main()
