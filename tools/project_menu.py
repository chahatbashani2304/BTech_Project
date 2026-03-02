import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "tests" / "images" / "dataset"
DB_PATH = PROJECT_ROOT / "attendance.db"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def prompt_non_empty(label: str) -> str:
    while True:
        value = input(label).strip()
        if value:
            return value
        print("Please enter a value.")


def add_person() -> None:
    print("\nAdd Person")
    name = prompt_non_empty("Enter person name (folder name): ")
    person_dir = DATASET_ROOT / name
    person_dir.mkdir(parents=True, exist_ok=True)
    print(f"Folder ready: {person_dir}")

    print("How do you want to add image?")
    print("1. Copy from existing file path")
    print("2. Capture from webcam")
    print("3. Skip for now")
    mode = input("Choose option (1/2/3): ").strip()

    if mode == "2":
        camera_index = input("Camera index (default 0): ").strip() or "0"
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "capture_person_image.py"),
                "--name",
                name,
                "--camera-backend",
                "auto",
                "--camera-index",
                camera_index,
            ]
        )
        return

    if mode == "3":
        print("Skipped image add. You can add .jpg/.jpeg/.png files later.")
        return

    if mode != "1":
        print("Invalid option.")
        return

    image_path_input = input("Enter image file path to copy: ").strip()
    source = Path(image_path_input).expanduser().resolve()
    if not source.exists() or not source.is_file():
        print(f"Image path not found: {source}")
        return
    if source.suffix.lower() not in VALID_IMAGE_EXTS:
        print("Unsupported image type. Use .jpg, .jpeg, or .png.")
        return

    destination = person_dir / source.name
    if destination.exists():
        destination = person_dir / f"{source.stem}_copy{source.suffix}"
    shutil.copy2(source, destination)
    print(f"Image copied: {destination}")


def run_command(cmd: list[str]) -> None:
    print("\nRunning:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}")


def update_attendance() -> None:
    print("\nUpdate Attendance")
    print("1. Scan dataset images")
    print("2. Realtime face scan (auto-identify)")
    sub = input("Choose option (1/2): ").strip()

    if sub == "1":
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "attendance_from_images.py"),
                "--images-root",
                str(DATASET_ROOT),
                "--db-path",
                str(DB_PATH),
                "--confidence-threshold",
                "0.9",
            ]
        )
        return

    if sub == "2":
        camera_index = input("Camera index (default 0): ").strip() or "0"
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "realtime_attendance.py"),
                "--db-path",
                str(DB_PATH),
                "--camera-backend",
                "auto",
                "--camera-index",
                camera_index,
                "--confidence-threshold",
                "0.9",
            ]
        )
        return

    print("Invalid option.")


def main() -> None:
    while True:
        print("\n=== Attendance Project Menu ===")
        print("1. Add Person")
        print("2. Update Attendance")
        print("3. Exit")
        choice = input("Choose option (1/2/3): ").strip()

        if choice == "1":
            add_person()
        elif choice == "2":
            update_attendance()
        elif choice == "3":
            print("Exiting.")
            return
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
