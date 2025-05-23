import sys

import cv2

from bot.puzzles.solver import PuzzleSolver
from tests.test_puzzle_solver import create_test_image


def visualize_test_image():
    """Create and visualize test images with different piece counts."""
    for n_pieces in [4, 9, 16]:
        print(f"\nCreating test image with {n_pieces} pieces...")
        test_image = create_test_image(n_pieces=n_pieces)
        solver = PuzzleSolver(test_image, expected_pieces=n_pieces)
        solver._save_debug_image(f"{n_pieces}_original", test_image)
        pieces = solver.detect_pieces()
        print(f"Detected {len(pieces)} pieces")
        try:
            solution = solver.solve()
            print(f"Solution matrix shape: {solution.shape}")
            print("Solution matrix:")
            print(solution)
        except Exception as e:
            print(f"Error solving puzzle: {e}")


def visualize_real_image(image_path, expected_pieces=0):
    """Visualize a real puzzle image from file."""
    test_image = cv2.imread(image_path)
    if test_image is None:
        print(f"Failed to load image: {image_path}")
        return
    print(f"\nLoaded real image: {image_path} shape={test_image.shape}")
    solver = PuzzleSolver(test_image, expected_pieces=expected_pieces)
    solver._save_debug_image("real_image_original", test_image)
    pieces = solver.detect_pieces()
    print(f"Detected {len(pieces)} pieces")
    try:
        solution = solver.solve()
        print(f"Solution matrix shape: {solution.shape}")
        print("Solution matrix:")
        print(solution)
    except Exception as e:
        print(f"Error solving puzzle: {e}")


if __name__ == "__main__":
    # Usage: python scripts/visualize_test.py [real_image_path expected_pieces]
    if len(sys.argv) == 3:
        image_path = sys.argv[1]
        expected_pieces = int(sys.argv[2])
        visualize_real_image(image_path, expected_pieces)
    else:
        visualize_test_image()
