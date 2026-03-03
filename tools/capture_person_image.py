import argparse
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "tests" / "images" / "dataset"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a person's image from webcam and save it into dataset folder."
    )
    parser.add_argument("--name", required=True, help="Person name (dataset folder).")
    parser.add_argument("--camera-index", type=int, default=0, help="Preferred camera index.")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "avfoundation", "any"],
        default="auto",
        help="OpenCV camera backend to use.",
    )
    args = parser.parse_args()

    person_dir = DATASET_ROOT / args.name.strip()
    person_dir.mkdir(parents=True, exist_ok=True)

    cap, active_index, active_backend = open_camera(args.camera_index, args.camera_backend)
    print(f"Camera opened: index={active_index}, backend={active_backend}")
    print("Controls: press 'c' to capture image, 'q' to quit.")

    saved_path = None
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Warning: could not read frame from camera.")
                continue

            cv2.putText(
                frame_bgr,
                f"{args.name} | c=capture, q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Capture Person Image", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                image_path = person_dir / f"{args.name}_{ts}.jpg"
                cv2.imwrite(str(image_path), frame_bgr)
                saved_path = image_path
                print(f"Saved: {image_path}")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if saved_path is None:
        print("No image captured.")


if __name__ == "__main__":
    main()
