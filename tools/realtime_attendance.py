import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

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
        INSERT INTO attendance_logs(student_id, image_path, timestamp, confidence)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(student_id, date(timestamp))
        DO UPDATE SET
            image_path = excluded.image_path,
            timestamp = excluded.timestamp,
            confidence = excluded.confidence
        """,
        (student_id, image_path, now_iso, confidence),
    )


def attendance_exists_today(cur: sqlite3.Cursor, student_id: int, now_iso: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM attendance_logs
        WHERE student_id = ? AND date(timestamp) = date(?)
        LIMIT 1
        """,
        (student_id, now_iso),
    )
    return cur.fetchone() is not None


def create_mtcnn_detector(device: str) -> MTCNN:
    try:
        return MTCNN(device=device)
    except TypeError:
        return MTCNN()


def clamp_box(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height
    return x1, y1, x2, y2


def backend_candidates(name: str) -> list[int]:
    if name == "avfoundation":
        return [cv2.CAP_AVFOUNDATION]
    if name == "any":
        return [cv2.CAP_ANY]
    return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]


def open_camera(camera_index: int, backend_name: str) -> tuple[cv2.VideoCapture, int, int]:
    index_candidates = [camera_index]
    for idx in (0, 1, 2):
        if idx not in index_candidates:
            index_candidates.append(idx)

    for idx in index_candidates:
        for backend in backend_candidates(backend_name):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(20):
                ok, _ = cap.read()
                if ok:
                    return cap, idx, backend
                time.sleep(0.05)
            cap.release()
    raise RuntimeError(
        "Could not open a working camera stream. "
        "Check macOS camera permissions and ensure no other app is using the camera."
    )


def iter_dataset_images(dataset_root: Path):
    allowed = {".jpg", ".jpeg", ".png"}
    if not dataset_root.exists():
        return
    for person_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        for img in person_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in allowed:
                yield person_dir.name, img


def face_embedding(image_rgb: np.ndarray, det: dict) -> Optional[np.ndarray]:
    h, w = image_rgb.shape[:2]
    box = det.get("box", [0, 0, 0, 0])
    x, y, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1, x2, y2 = clamp_box(x, y, bw, bh, w, h)
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    emb = gray.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    if norm == 0.0:
        return None
    return emb / norm


def build_face_database(
    detector: MTCNN,
    dataset_root: Path,
    detection_conf_threshold: float,
) -> dict[str, np.ndarray]:
    by_person: dict[str, list[np.ndarray]] = {}
    for person_name, img_path in iter_dataset_images(dataset_root):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            dets = detector.detect_faces(rgb)
        except Exception:
            continue
        if not dets:
            continue
        best = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
        conf = float(best.get("confidence", 0.0))
        if conf < detection_conf_threshold:
            continue
        emb = face_embedding(rgb, best)
        if emb is None:
            continue
        by_person.setdefault(person_name, []).append(emb)

    averaged: dict[str, np.ndarray] = {}
    for person_name, embs in by_person.items():
        vec = np.mean(np.stack(embs, axis=0), axis=0)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            averaged[person_name] = vec / norm
    return averaged


def identify_face(emb: np.ndarray, face_db: dict[str, np.ndarray], threshold: float) -> tuple[str, float]:
    best_name = "unknown"
    best_sim = -1.0
    for name, ref in face_db.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim < threshold:
        return "unknown", best_sim
    return best_name, best_sim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-time webcam attendance. If --name is omitted, auto-identify from dataset."
    )
    parser.add_argument("--name", default=None, help="Optional fixed name for manual marking mode.")
    parser.add_argument("--db-path", type=Path, default=Path("attendance.db"), help="SQLite database path.")
    parser.add_argument("--device", default="CPU:0", help='MTCNN device, e.g. "CPU:0" or "CUDA:0".')
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for OpenCV.")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "avfoundation", "any"],
        default="auto",
        help="OpenCV camera backend to use (recommended on macOS: avfoundation).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Only mark attendance when face confidence is >= this threshold.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("tests/images/lpw_small"),
        help="Dataset root used for auto-identification mode.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for auto identity match.",
    )
    parser.add_argument(
        "--auto-mark-cooldown-seconds",
        type=float,
        default=5.0,
        help="Minimum seconds between repeated auto-mark attempts for same identity.",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path("tests/images/realtime"),
        help="Where realtime captured attendance frames are stored.",
    )
    args = parser.parse_args()

    detector = create_mtcnn_detector(args.device)
    conn = init_db(args.db_path.resolve())
    cur = conn.cursor()
    dedupe_and_enforce_daily_uniqueness(cur)
    conn.commit()

    manual_mode = args.name is not None and args.name.strip() != ""
    fixed_name = args.name.strip() if manual_mode else None
    face_db: dict[str, np.ndarray] = {}

    if manual_mode:
        get_or_create_student(cur, fixed_name)
        conn.commit()
        print(f"Realtime mode: manual ({fixed_name})")
    else:
        face_db = build_face_database(
            detector=detector,
            dataset_root=args.dataset_root.resolve(),
            detection_conf_threshold=args.confidence_threshold,
        )
        if not face_db:
            conn.close()
            raise RuntimeError(
                "Auto-identification database is empty. Add clear face images under "
                "tests/images/lpw_small/<PersonName>/ and retry."
            )
        print(f"Realtime mode: auto-identify ({len(face_db)} identities loaded)")

    cap, active_index, active_backend = open_camera(args.camera_index, args.camera_backend)

    print("Realtime attendance started.")
    print(f"Camera opened: index={active_index}, backend={active_backend}")
    if manual_mode:
        print("Controls: press 's' to save attendance, 'q' to quit.")
    else:
        print("Controls: auto-mark is enabled; press 'q' to quit.")

    failed_reads = 0
    last_auto_mark: dict[str, float] = {}

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                failed_reads += 1
                print("Warning: could not read frame from camera.")
                if failed_reads >= 20:
                    cap.release()
                    cap, active_index, active_backend = open_camera(args.camera_index, args.camera_backend)
                    print(f"Camera reconnected: index={active_index}, backend={active_backend}")
                    failed_reads = 0
                continue
            failed_reads = 0

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                detections = detector.detect_faces(frame_rgb)
            except Exception as exc:
                detections = []
                print(f"Warning: detection failed for frame ({exc}).")

            h, w = frame_bgr.shape[:2]
            max_conf = 0.0
            best_name = "unknown"
            best_sim = -1.0

            for det in detections:
                conf = float(det.get("confidence", 0.0))
                max_conf = max(max_conf, conf)

                label_name = fixed_name if manual_mode else "unknown"
                sim = -1.0

                if not manual_mode and conf >= args.confidence_threshold:
                    emb = face_embedding(frame_rgb, det)
                    if emb is not None:
                        label_name, sim = identify_face(emb, face_db, args.match_threshold)
                        if sim > best_sim:
                            best_sim = sim
                            best_name = label_name

                box = det.get("box", [0, 0, 0, 0])
                x, y, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x1, y1, x2, y2 = clamp_box(x, y, bw, bh, w, h)

                if manual_mode:
                    color = (0, 255, 0) if conf >= args.confidence_threshold else (0, 165, 255)
                    txt = f"{label_name} conf={conf:.3f}"
                else:
                    is_match = label_name != "unknown"
                    color = (0, 255, 0) if is_match else (0, 165, 255)
                    txt = f"{label_name} c={conf:.3f} s={sim:.3f}"

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_bgr,
                    txt,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            faces_detected = len(detections)
            if manual_mode:
                status = (
                    f"{fixed_name} | faces={faces_detected} | max_conf={max_conf:.3f} | "
                    "press s=mark attendance, q=quit"
                )
            else:
                status = (
                    f"auto-identify | faces={faces_detected} | best={best_name} | "
                    f"sim={best_sim:.3f} | q=quit"
                )

            cv2.putText(frame_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.imshow("Realtime Attendance", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            now_iso = datetime.now().isoformat(timespec="seconds")

            if not manual_mode:
                if (
                    faces_detected > 0
                    and max_conf >= args.confidence_threshold
                    and best_name != "unknown"
                    and best_sim >= args.match_threshold
                ):
                    last_t = last_auto_mark.get(best_name, 0.0)
                    now_t = time.time()
                    if now_t - last_t >= args.auto_mark_cooldown_seconds:
                        student_id = get_or_create_student(cur, best_name)
                        if not attendance_exists_today(cur, student_id, now_iso):
                            capture_dir = (args.save_root / best_name).resolve()
                            capture_dir.mkdir(parents=True, exist_ok=True)
                            filename = f"{best_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            image_path = capture_dir / filename
                            cv2.imwrite(str(image_path), frame_bgr)

                            cur.execute(
                                """
                                INSERT INTO scans(student_id, image_path, faces_detected, scanned_at)
                                VALUES (?, ?, ?, ?)
                                """,
                                (student_id, str(image_path), faces_detected, now_iso),
                            )
                            upsert_daily_attendance(cur, student_id, str(image_path), now_iso, max_conf)
                            conn.commit()
                            print(
                                f"Attendance auto-marked for {best_name} at {now_iso} "
                                f"(conf={max_conf:.3f}, sim={best_sim:.3f})"
                            )
                        last_auto_mark[best_name] = now_t
                continue

            if key == ord("s"):
                student_id = get_or_create_student(cur, fixed_name)
                capture_dir = (args.save_root / fixed_name).resolve()
                capture_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{fixed_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = capture_dir / filename
                cv2.imwrite(str(image_path), frame_bgr)

                cur.execute(
                    """
                    INSERT INTO scans(student_id, image_path, faces_detected, scanned_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (student_id, str(image_path), faces_detected, now_iso),
                )

                if faces_detected > 0 and max_conf >= args.confidence_threshold:
                    upsert_daily_attendance(cur, student_id, str(image_path), now_iso, max_conf)
                    print(
                        f"Attendance marked for {fixed_name} at {now_iso} "
                        f"(faces={faces_detected}, conf={max_conf:.3f})"
                    )
                else:
                    print(
                        "Scan saved but attendance not marked: "
                        f"faces={faces_detected}, max_conf={max_conf:.3f}, "
                        f"threshold={args.confidence_threshold:.3f}"
                    )
                conn.commit()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()


if __name__ == "__main__":
    main()
