import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from mtcnn import MTCNN


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    f1_score: float
    recall: float
    precision: float
    tp: int
    tn: int
    fp: int
    fn: int
    total_samples: int
    evaluated_at: str


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for ext in IMAGE_EXTS:
        yield from path.rglob(ext)


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detector_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL NOT NULL,
            f1_score REAL NOT NULL,
            recall REAL NOT NULL,
            precision REAL NOT NULL,
            tp INTEGER NOT NULL,
            tn INTEGER NOT NULL,
            fp INTEGER NOT NULL,
            fn INTEGER NOT NULL,
            total_samples INTEGER NOT NULL,
            evaluated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    total = tp + tn + fp + fn
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    return accuracy, f1_score, recall, precision


def mtcnn_detector_factory(device: str, conf_threshold: float) -> Callable[[Path], bool]:
    detector = MTCNN(device=device)

    def detect(image_path: Path) -> bool:
        detections = detector.detect_faces(str(image_path))
        max_conf = max((float(d.get("confidence", 0.0)) for d in detections), default=0.0)
        return len(detections) > 0 and max_conf >= conf_threshold

    return detect


def retinaface_detector_factory(conf_threshold: float) -> Callable[[Path], bool]:
    try:
        from retinaface import RetinaFace  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RetinaFace not available. Install with: pip install retina-face"
        ) from exc

    def detect(image_path: Path) -> bool:
        result = RetinaFace.detect_faces(str(image_path))
        if not isinstance(result, dict) or len(result) == 0:
            return False
        scores = [
            float(v.get("score", 0.0))
            for v in result.values()
            if isinstance(v, dict)
        ]
        return max(scores, default=0.0) >= conf_threshold

    return detect


def yolo_detector_factory(model_path: str, conf_threshold: float) -> Callable[[Path], bool]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "YOLO not available. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(model_path)

    def detect(image_path: Path) -> bool:
        result = model(str(image_path), verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return False
        confs = result.boxes.conf.tolist() if result.boxes.conf is not None else []
        return max((float(c) for c in confs), default=0.0) >= conf_threshold

    return detect


def evaluate_model(
    model_name: str,
    detect_fn: Callable[[Path], bool],
    positive_images: list[Path],
    negative_images: list[Path],
) -> EvalResult:
    tp = tn = fp = fn = 0
    for image_path in positive_images:
        pred_face = detect_fn(image_path)
        if pred_face:
            tp += 1
        else:
            fn += 1
    for image_path in negative_images:
        pred_face = detect_fn(image_path)
        if pred_face:
            fp += 1
        else:
            tn += 1

    accuracy, f1_score, recall, precision = compute_metrics(tp, tn, fp, fn)
    return EvalResult(
        model_name=model_name,
        accuracy=accuracy,
        f1_score=f1_score,
        recall=recall,
        precision=precision,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total_samples=tp + tn + fp + fn,
        evaluated_at=datetime.now().isoformat(timespec="seconds"),
    )


def save_results(conn: sqlite3.Connection, results: list[EvalResult], reset_today: bool) -> None:
    cur = conn.cursor()
    today = datetime.now().date().isoformat()
    if reset_today:
        cur.execute("DELETE FROM detector_metrics WHERE date(evaluated_at) = ?", (today,))

    for r in results:
        cur.execute(
            """
            INSERT INTO detector_metrics(
                model_name, accuracy, f1_score, recall, precision,
                tp, tn, fp, fn, total_samples, evaluated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.model_name,
                r.accuracy,
                r.f1_score,
                r.recall,
                r.precision,
                r.tp,
                r.tn,
                r.fp,
                r.fn,
                r.total_samples,
                r.evaluated_at,
            ),
        )
    conn.commit()


def print_results(results: list[EvalResult]) -> None:
    print("-" * 88)
    print(f"{'model':12} {'accuracy':10} {'f1_score':10} {'recall':10} {'precision':10} {'tp':5} {'fp':5} {'fn':5}")
    print("-" * 88)
    for r in results:
        print(
            f"{r.model_name:12} {r.accuracy:<10.4f} {r.f1_score:<10.4f} "
            f"{r.recall:<10.4f} {r.precision:<10.4f} {r.tp:<5} {r.fp:<5} {r.fn:<5}"
        )
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MTCNN, RetinaFace, and YOLO on face/non-face images and store metrics."
    )
    parser.add_argument("--db-path", type=Path, default=Path("attendance.db"))
    parser.add_argument("--positive-root", type=Path, default=Path("tests/images/lpw_small"))
    parser.add_argument(
        "--negative-root",
        type=Path,
        default=Path("tests/images/no-faces.jpg"),
        help="Folder or file with non-face images.",
    )
    parser.add_argument("--device", default="CPU:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--yolo-model", default="yolov8n-face.pt")
    parser.add_argument(
        "--no-reset-today",
        action="store_true",
        help="Do not clear today's detector metrics before inserting new results.",
    )
    args = parser.parse_args()

    positive_root = args.positive_root.resolve()
    negative_root = args.negative_root.resolve()
    if not positive_root.exists():
        raise FileNotFoundError(f"Positive root not found: {positive_root}")
    if not negative_root.exists():
        raise FileNotFoundError(f"Negative root not found: {negative_root}")

    positive_images = sorted(iter_images(positive_root))
    negative_images = sorted(iter_images(negative_root))
    if not positive_images:
        raise RuntimeError(f"No positive images found in {positive_root}")
    if not negative_images:
        raise RuntimeError(f"No negative images found in {negative_root}")

    model_factories: list[tuple[str, Callable[[], Callable[[Path], bool]]]] = [
        ("mtcnn", lambda: mtcnn_detector_factory(args.device, args.confidence_threshold)),
        ("retinaface", lambda: retinaface_detector_factory(args.confidence_threshold)),
        ("yolo", lambda: yolo_detector_factory(args.yolo_model, args.confidence_threshold)),
    ]

    results: list[EvalResult] = []
    for model_name, factory in model_factories:
        try:
            detector_fn = factory()
        except Exception as exc:
            print(f"Skipping {model_name}: {exc}")
            continue
        result = evaluate_model(model_name, detector_fn, positive_images, negative_images)
        results.append(result)

    if not results:
        raise RuntimeError("No models were evaluated. Install RetinaFace/YOLO dependencies and retry.")

    conn = init_db(args.db_path.resolve())
    save_results(conn, results, reset_today=not args.no_reset_today)
    conn.close()
    print_results(results)
    print(f"Saved metrics in table: detector_metrics ({args.db_path.resolve()})")


if __name__ == "__main__":
    main()
