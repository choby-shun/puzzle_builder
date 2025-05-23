import cv2
import numpy as np
import pytest

from bot.puzzles.solver import PuzzlePiece, PuzzleSolver


@pytest.fixture
def basic_test_image():
    """Fixture for creating a basic test image with 4 pieces."""
    return create_test_image(n_pieces=4)


@pytest.fixture
def complex_test_image():
    """Fixture for creating a complex test image with 9 pieces."""
    return create_complex_test_image(n_pieces=9)


@pytest.fixture
def basic_solver(basic_test_image):
    """Fixture for creating a basic puzzle solver instance."""
    return PuzzleSolver(basic_test_image, expected_pieces=4)


@pytest.fixture
def complex_solver(complex_test_image):
    """Fixture for creating a complex puzzle solver instance."""
    return PuzzleSolver(complex_test_image, expected_pieces=9)


def create_test_image(size=(500, 500), n_pieces=4):
    """Create a test image with n_pieces pieces."""
    # Create a black image (background)
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Calculate piece dimensions
    grid_size = int(np.sqrt(n_pieces))
    piece_width = size[0] // grid_size
    piece_height = size[1] // grid_size

    # Create each piece as a white rectangle with a small gap
    gap = 2  # Gap between pieces
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate piece coordinates
            x1 = j * piece_width + gap
            y1 = i * piece_height + gap
            x2 = (j + 1) * piece_width - gap
            y2 = (i + 1) * piece_height - gap

            # Draw white rectangle for the piece
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

    return image


def create_complex_test_image(n_pieces: int = 9) -> np.ndarray:
    """Create a more complex test image with interlocking puzzle pieces.

    Args:
        n_pieces: Number of pieces to create (default: 9)

    Returns:
        np.ndarray: Test image with interlocking puzzle pieces
    """
    # Create a black background
    image = np.zeros((800, 800, 3), dtype=np.uint8)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_pieces)))
    piece_width = image.shape[1] // grid_size
    piece_height = image.shape[0] // grid_size

    # Create pieces with interlocking shapes
    for i in range(n_pieces):
        row = i // grid_size
        col = i % grid_size

        # Base position for this piece
        x = col * piece_width
        y = row * piece_height

        # Create piece shape with interlocking edges
        points = []

        # Define the base rectangle
        base_points = [
            [x, y],  # Top-left
            [x + piece_width, y],  # Top-right
            [x + piece_width, y + piece_height],  # Bottom-right
            [x, y + piece_height],  # Bottom-left
        ]

        # Add interlocking shapes to each edge
        for j in range(4):
            # Get the current edge points
            p1 = base_points[j]
            p2 = base_points[(j + 1) % 4]

            # Calculate edge direction and length
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            edge_length = max(abs(dx), abs(dy))

            # Add the first point of the edge
            points.append(p1)

            # Add interlocking shape
            if edge_length > 0:
                # Calculate number of interlocking points
                num_points = 3
                for k in range(1, num_points):
                    # Calculate position along the edge
                    t = k / num_points
                    px = p1[0] + dx * t
                    py = p1[1] + dy * t

                    # Add random offset perpendicular to the edge
                    if abs(dx) > abs(dy):  # Horizontal edge
                        offset = np.random.randint(
                            -piece_height // 8, piece_height // 8
                        )
                        points.append([px, py + offset])
                    else:  # Vertical edge
                        offset = np.random.randint(-piece_width // 8, piece_width // 8)
                        points.append([px + offset, py])

        # Convert points to numpy array and draw the piece
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(image, [points], (255, 255, 255))

        # Add some texture to the piece
        for _ in range(3):
            center_x = (
                x
                + piece_width // 2
                + np.random.randint(-piece_width // 4, piece_width // 4)
            )
            center_y = (
                y
                + piece_height // 2
                + np.random.randint(-piece_height // 4, piece_height // 4)
            )
            radius = np.random.randint(5, 10)
            cv2.circle(image, (center_x, center_y), radius, (200, 200, 200), -1)

    return image


class TestPuzzleSolverInitialization:
    """Test cases for puzzle solver initialization."""

    def test_valid_initialization(self, basic_test_image):
        """Test puzzle solver initialization with valid input."""
        solver = PuzzleSolver(basic_test_image, expected_pieces=4)
        assert solver.image is not None
        assert len(solver.pieces) == 0
        assert solver.expected_pieces == 4
        assert not np.shares_memory(solver.image, basic_test_image)

    def test_invalid_input(self, basic_test_image):
        """Test puzzle solver initialization with invalid inputs."""
        # Test with invalid numpy array (2D instead of 3D)
        with pytest.raises(ValueError):
            PuzzleSolver(np.zeros((100, 100)), expected_pieces=4)

        # Test with invalid expected_pieces
        with pytest.raises(TypeError):
            PuzzleSolver(basic_test_image, expected_pieces="4")  # type: ignore
        with pytest.raises(TypeError):
            PuzzleSolver(basic_test_image, expected_pieces=0)
        with pytest.raises(TypeError):
            PuzzleSolver(basic_test_image, expected_pieces=-1)


class TestPieceDetection:
    """Test cases for piece detection functionality."""

    def test_basic_piece_detection(self, basic_solver):
        """Test basic piece detection with simple pieces."""
        pieces = basic_solver.detect_pieces()
        assert len(pieces) == 4

        for piece in pieces:
            assert isinstance(piece, PuzzlePiece)
            assert piece.id >= 0
            assert len(piece.corners) >= 3
            assert isinstance(piece.center, tuple)
            assert len(piece.center) == 2
            assert isinstance(piece.image, np.ndarray)
            assert piece.area > 0

    def test_complex_piece_detection(self, complex_solver):
        """Test piece detection with interlocking pieces."""
        pieces = complex_solver.detect_pieces()
        assert len(pieces) == 9, f"Expected 9 pieces, got {len(pieces)}"

        # Verify each piece has valid properties
        for piece in pieces:
            assert len(piece.corners) >= 3, f"Piece {piece.id} has less than 3 corners"
            assert piece.area > 0, f"Piece {piece.id} has zero area"
            assert (
                piece.center[0] >= 0 and piece.center[1] >= 0
            ), f"Piece {piece.id} has invalid center coordinates"
            assert (
                piece.image.shape[:2] == complex_solver.image.shape[:2]
            ), f"Piece {piece.id} has incorrect image dimensions"

        # Verify pieces don't overlap significantly
        for i, piece1 in enumerate(pieces):
            for j, piece2 in enumerate(pieces[i + 1 :], i + 1):
                # Calculate intersection area
                mask1 = np.zeros(complex_solver.image.shape[:2], dtype=np.uint8)
                mask2 = np.zeros(complex_solver.image.shape[:2], dtype=np.uint8)

                cv2.fillPoly(mask1, [np.array(piece1.corners, dtype=np.int32)], 255)
                cv2.fillPoly(mask2, [np.array(piece2.corners, dtype=np.int32)], 255)

                intersection = cv2.bitwise_and(mask1, mask2)
                intersection_area = np.count_nonzero(intersection)

                # Allow small overlap due to anti-aliasing
                max_allowed_overlap = min(piece1.area, piece2.area) * 0.05
                assert (
                    intersection_area <= max_allowed_overlap
                ), f"Pieces {i} and {j} overlap too much: {intersection_area} pixels"

    def test_piece_validation(self, basic_test_image):
        """Test piece validation with overlapping pieces."""
        # Draw an overlapping piece
        cv2.rectangle(basic_test_image, (100, 100), (300, 300), (0, 0, 0), 2)
        solver = PuzzleSolver(basic_test_image, expected_pieces=4)

        with pytest.raises(ValueError):
            solver.detect_pieces()

    def test_wrong_number_of_pieces(self, basic_test_image):
        """Test validation when wrong number of pieces is detected."""
        solver = PuzzleSolver(basic_test_image, expected_pieces=9)

        with pytest.raises(ValueError) as exc_info:
            solver.detect_pieces()
        assert "Expected 9 pieces, but found 4" in str(exc_info.value)


class TestPuzzleSolving:
    """Test cases for puzzle solving functionality."""

    def test_basic_solve(self, basic_solver):
        """Test solving a basic 4-piece puzzle."""
        basic_solver.detect_pieces()
        solution = basic_solver.solve()

        assert isinstance(solution, np.ndarray)
        assert solution.shape == (2, 2)  # For 4 pieces, should be 2x2
        assert set(solution.flatten()) == set(range(4))  # Should contain all piece IDs

    def test_complex_solve(self, complex_solver):
        """Test solving a complex 9-piece puzzle."""
        complex_solver.detect_pieces()
        solution = complex_solver.solve()

        assert isinstance(solution, np.ndarray)
        assert solution.shape == (3, 3)  # For 9 pieces, should be 3x3
        assert set(solution.flatten()) == set(range(9))  # Should contain all piece IDs
