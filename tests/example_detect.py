# from mtcnn import MTCNN
# from mtcnn.utils.images import load_image
# import pprint
# from pathlib import Path
# import sys


# def main():
# 	# Create detector
# 	detector = MTCNN(device="CPU:0")

# 	# Build a cross-platform path to the example image located in the project tests directory
# 	# File location (project root)/tests/images/ivan.jpg
# 	script_path = Path(__file__).resolve()
# 	# script_path is .../mtcnn/mtcnn/example_detect.py, project root is two parents up
# 	project_root = script_path.parents[1]
# 	image_path = project_root / "tests" / "images" / "ivan.jpg"

# 	if not image_path.exists():
# 		print(f"Image not found at {image_path}")
# 		sys.exit(1)

# 	# Load image
# 	image = load_image(str(image_path))

# 	# Detect faces
# 	result = detector.detect_faces(image)

# 	# Print results
# 	pprint.pprint(result)


# if __name__ == "__main__":
# 	main()


from mtcnn import MTCNN
from mtcnn.utils.images import load_image
import pprint
from pathlib import Path
import sys

def iter_image_paths(default_dir: Path, cli_args: list[str]) -> list[Path]:
    if cli_args:
        paths = [Path(p).resolve() for p in cli_args]
    else:
        # default: all images in tests/images/
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(sorted(default_dir.glob(ext)))
    return [p for p in paths if p.exists()]

def main():
    # Create detector (CPU; change to "CUDA:0" if you have a GPU)
    detector = MTCNN(device="CPU:0")

    # Build project-rooted default images directory: (project root)/tests/images/
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    default_images_dir = project_root / "tests" / "images"

    # Collect image paths (CLI args override the default folder)
    image_paths = iter_image_paths(default_images_dir, sys.argv[1:])
    if not image_paths:
        print(f"No images found. Put images in {default_images_dir} "
              f"or pass file paths:  python {script_path.name} path/to/a.jpg path/to/b.png")
        sys.exit(1)

    # Process each image
    for img_path in image_paths:
        print(f"\n=== {img_path} ===")
        image = load_image(str(img_path))
        result = detector.detect_faces(image)
        pprint.pprint(result)

if __name__ == "__main__":
    main()

