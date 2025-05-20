from bot.puzzles.solver import PuzzleSolver
from tests.test_puzzle_solver import create_test_image


def visualize_test_image():
    """Create and visualize test images with different piece counts."""
    # Test different piece counts
    for n_pieces in [4, 9, 16]:
        print(f"\nCreating test image with {n_pieces} pieces...")

        # Create test image
        test_image = create_test_image(n_pieces=n_pieces)

        # Initialize solver with debug enabled
        solver = PuzzleSolver(test_image, expected_pieces=n_pieces, debug=True)

        # Save original test image
        solver._save_debug_image(f"{n_pieces}_original", test_image)

        # Detect pieces
        pieces = solver.detect_pieces()
        print(f"Detected {len(pieces)} pieces")

        # Try to solve
        try:
            solution = solver.solve()
            print(f"Solution matrix shape: {solution.shape}")
            print("Solution matrix:")
            print(solution)
        except Exception as e:
            print(f"Error solving puzzle: {e}")


if __name__ == "__main__":
    visualize_test_image()
