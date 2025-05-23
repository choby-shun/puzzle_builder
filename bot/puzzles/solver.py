import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import PuzzlePiece

DEBUG = os.environ.get("DEBUG", True)


class PuzzleSolver:
    """Main class for solving jigsaw puzzles using computer vision."""

    def __init__(self, image: np.ndarray, expected_pieces: int = 0):
        """Initialize the puzzle solver with an image and expected number of pieces.

        Args:
            image: A numpy array containing the image data in BGR format (OpenCV default).
                  Must be a 3-channel image.
            expected_pieces: The expected number of pieces in the puzzle.
            debug: If True, enables debug visualizations and logging.

        Raises:
            ValueError: If the image is not a valid 3-channel BGR image
            TypeError: If expected_pieces is not a positive integer
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR image")

        if expected_pieces == 0:
            expected_pieces = len(
                cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            )

        self.image = image.copy()  # Make a copy to avoid modifying the original
        self.expected_pieces = expected_pieces
        self.pieces: List[PuzzlePiece] = []

        if DEBUG:
            # Create timestamped debug directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = Path("debug_images") / timestamp
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"\nDebug session: {timestamp}")
            print(f"Debug directory: {self.debug_dir}")

    def _print_image_info(self) -> None:
        """Print basic information about the image."""
        print("\nImage Information:")
        print(f"Shape: {self.image.shape}")
        print(f"Type: {self.image.dtype}")
        print(f"Min value: {np.min(self.image)}")
        print(f"Max value: {np.max(self.image)}")
        print(f"Mean value: {np.mean(self.image):.2f}")
        print(f"Expected pieces: {self.expected_pieces}\n")

    def _save_debug_image(self, name: str, image: np.ndarray) -> None:
        """Save a debug image with timestamp.

        Args:
            name: Name of the debug image
            image: Image to save
        """
        if not DEBUG:
            return

        # Save image with timestamp
        filename = self.debug_dir / f"{name}.png"
        cv2.imwrite(str(filename), image)
        print(f"Saved debug image: {filename}")

    def _draw_pieces(self, image: np.ndarray) -> np.ndarray:
        """Draw detected pieces on the image.

        Args:
            image: Image to draw on

        Returns:
            Image with pieces drawn
        """
        debug_image = image.copy()

        # Draw each piece
        for piece in self.pieces:
            # Draw piece boundary
            corners = np.array(piece.corners, dtype=np.int32)
            cv2.polylines(debug_image, [corners], True, (0, 255, 0), 2)

            # Draw piece center
            cv2.circle(debug_image, piece.center, 5, (0, 0, 255), -1)

            # Draw piece ID
            cv2.putText(
                debug_image,
                str(piece.id),
                piece.center,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        return debug_image

    def detect_pieces(self) -> List[PuzzlePiece]:
        """Detect and extract individual puzzle pieces from the image.

        Returns:
            List of detected puzzle pieces

        Raises:
            ValueError: If pieces overlap or if there are missing pieces
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if DEBUG:
            self._save_debug_image("02_grayscale", gray)

        # Apply threshold to get binary image
        # Use a fixed threshold since we know pieces are white (255) on black (0)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        if DEBUG:
            self._save_debug_image("03_binary", binary)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each contour into a puzzle piece
        for i, contour in enumerate(contours):
            # Simplify the contour while preserving the shape
            # Reduced from 0.02 to preserve more detail
            epsilon = 0.005 * cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, epsilon, True)

            # Calculate center using moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Calculate area
            area = cv2.contourArea(contour)

            # Create piece object
            piece = PuzzlePiece(
                id=i,
                corners=corners.reshape(-1, 2).tolist(),
                center=(cx, cy),
                image=self._extract_piece(contour),
                area=area,
            )
            self.pieces.append(piece)

        # Validate pieces
        self._validate_pieces()

        # Save debug image with detected pieces
        if DEBUG:
            debug_image = self._draw_pieces(self.image)
            self._save_debug_image("04_detected_pieces", debug_image)

        return self.pieces

    def _validate_pieces(self) -> None:
        """Validate that pieces are correctly detected.

        Raises:
            ValueError: If validation fails
        """
        if not self.pieces:
            raise ValueError("No pieces detected")

        # Check if we found the expected number of pieces
        if len(self.pieces) != self.expected_pieces:
            raise ValueError(
                f"Expected {self.expected_pieces} pieces, but found {len(self.pieces)}"
            )

        # Check for minimum number of corners (at least 3 for a valid polygon)
        for i, piece in enumerate(self.pieces):
            if len(piece.corners) < 3:
                raise ValueError(f"Piece {i} has less than 3 corners")

    def _extract_piece(self, contour: np.ndarray) -> np.ndarray:
        """Extract the image of a single puzzle piece.

        Args:
            contour: Contour of the piece

        Returns:
            Image of the piece
        """
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Ensure proper format for cv2.fillPoly
        contour = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)

        cv2.fillPoly(mask, [contour], 255)  # type: ignore[arg-type]

        # Apply mask
        piece = cv2.bitwise_and(self.image, self.image, mask=mask)

        return piece

    def solve(self) -> np.ndarray:
        """Solve the puzzle and return the solution matrix.

        Returns:
            Matrix where each element is the ID of the piece in that position
        """
        if not self.pieces:
            raise ValueError("No pieces detected. Call detect_pieces() first.")

        # Determine puzzle dimensions based on piece analysis
        rows, cols = self._determine_puzzle_dimensions()
        self.solution_matrix: np.ndarray = np.full((rows, cols), -1, dtype=np.int32)

        # First, identify corner pieces
        corners = self._find_corner_pieces()
        if len(corners) != 4:
            raise ValueError(f"Expected 4 corner pieces, found {len(corners)}")

        # Place corner pieces
        self._place_corner_pieces(self.solution_matrix, corners)

        # Build edges
        self._build_edges(self.solution_matrix)

        # Fill interior
        self._fill_interior()

        if DEBUG:
            self._visualize_solution()

        return self.solution_matrix

    def _determine_puzzle_dimensions(self) -> Tuple[int, int]:
        """Determine the number of rows and columns in the puzzle.

        This method analyzes the pieces to determine the most likely puzzle dimensions.
        It considers:
        1. The total number of pieces
        2. Common puzzle aspect ratios
        3. The shape of edge pieces

        Returns:
            Tuple of (rows, columns)
        """
        # Common puzzle aspect ratios (width:height)
        common_ratios = [
            (1, 1),  # Square
            (4, 3),  # Standard
            (16, 9),  # Widescreen
            (2, 1),  # Panoramic
            (3, 2),  # Classic
        ]

        # Find the best matching ratio
        best_ratio = None
        best_error = float("inf")
        total_pieces = len(self.pieces)

        for width, height in common_ratios:
            # Calculate the number of pieces that would fit this ratio
            # while maintaining approximately the total number of pieces
            scale = np.sqrt(total_pieces / (width * height))
            rows = int(round(height * scale))
            cols = int(round(width * scale))

            # Calculate how close we are to the target number of pieces
            error = abs(rows * cols - total_pieces)

            if error < best_error:
                best_error = error
                best_ratio = (rows, cols)

        if best_ratio is None:
            # Fallback to square if no good ratio found
            side = int(np.ceil(np.sqrt(total_pieces)))
            return (side, side)

        return best_ratio

    def _find_corner_pieces(self) -> List[PuzzlePiece]:
        """Find pieces that are likely to be corners.

        Returns:
            List of corner pieces
        """
        corners = []
        for piece in self.pieces:
            # Count straight edges (edges that are mostly horizontal or vertical)
            straight_edges = 0
            for i in range(len(piece.corners)):
                p1 = piece.corners[i]
                p2 = piece.corners[(i + 1) % len(piece.corners)]

                # Calculate edge angle
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

                # Check if edge is approximately horizontal or vertical
                if angle < 30 or angle > 150:
                    straight_edges += 1

            # Corner pieces have exactly 2 straight edges
            if straight_edges == 2:
                corners.append(piece)

        return corners

    def _place_corner_pieces(
        self, solution_matrix: np.ndarray, corners: List[PuzzlePiece]
    ) -> None:
        """Place corner pieces in their correct positions.

        Args:
            corners: List of corner pieces
        """
        # Sort corners by position (top-left, top-right, bottom-left, bottom-right)
        corners.sort(key=lambda p: (p.center[1], p.center[0]))

        # Place corners
        solution_matrix[0, 0] = corners[0].id  # Top-left
        solution_matrix[0, -1] = corners[1].id  # Top-right
        solution_matrix[-1, 0] = corners[2].id  # Bottom-left
        solution_matrix[-1, -1] = corners[3].id  # Bottom-right

    def _build_edges(self, solution_matrix: np.ndarray) -> None:
        """Build the puzzle edges by matching pieces."""
        rows, cols = solution_matrix.shape

        # Build top edge
        for col in range(1, cols - 1):
            self._find_and_place_edge_piece(0, col, "top")

        # Build bottom edge
        for col in range(1, cols - 1):
            self._find_and_place_edge_piece(rows - 1, col, "bottom")

        # Build left edge
        for row in range(1, rows - 1):
            self._find_and_place_edge_piece(row, 0, "left")

        # Build right edge
        for row in range(1, rows - 1):
            self._find_and_place_edge_piece(row, cols - 1, "right")

    def _find_and_place_edge_piece(self, row: int, col: int, edge_type: str) -> None:
        """Find and place an edge piece at the specified position.

        Args:
            row: Row position
            col: Column position
            edge_type: Type of edge ("top", "bottom", "left", "right")
        """
        # Get adjacent pieces
        adjacent_pieces = self._get_adjacent_pieces(row, col)

        # Find best matching piece
        best_piece = None
        best_score = float("-inf")

        for piece in self.pieces:
            if piece.id in self.solution_matrix:
                continue

            score = self._calculate_edge_match_score(piece, adjacent_pieces, edge_type)
            if score > best_score:
                best_score = score
                best_piece = piece

        if best_piece is not None:
            self.solution_matrix[row, col] = best_piece.id

    def _get_adjacent_pieces(self, row: int, col: int) -> List[Optional[PuzzlePiece]]:
        """Get pieces adjacent to the specified position.

        Args:
            row: Row position
            col: Column position

        Returns:
            List of adjacent pieces (None for empty positions)
        """
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < self.solution_matrix.shape[0]
                and 0 <= new_col < self.solution_matrix.shape[1]
            ):
                piece_id = self.solution_matrix[new_row, new_col]
                piece = next((p for p in self.pieces if p.id == piece_id), None)
                adjacent.append(piece)
            else:
                adjacent.append(None)
        return adjacent

    def _calculate_edge_match_score(
        self,
        piece: PuzzlePiece,
        adjacent_pieces: List[Optional[PuzzlePiece]],
        edge_type: str,
    ) -> np.float64:
        """Calculate how well a piece matches with its adjacent pieces.

        Args:
            piece: Piece to evaluate
            adjacent_pieces: List of adjacent pieces
            edge_type: Type of edge being matched

        Returns:
            Match score (higher is better)
        """
        score = 0.0

        # Check color matching at edges
        for adj_piece in adjacent_pieces:
            if adj_piece is not None:
                # Calculate color difference at shared edge
                color_diff = self._calculate_edge_color_difference(piece, adj_piece)
                score -= color_diff  # Lower difference is better

        # Check if piece has a straight edge on the correct side
        if self._has_straight_edge(piece, edge_type):
            score += 100

        return np.float64(score)

    def _calculate_edge_color_difference(
        self, piece1: PuzzlePiece, piece2: PuzzlePiece
    ) -> np.float64:
        """Calculate color difference between two pieces at their shared edge.

        Args:
            piece1: First piece
            piece2: Second piece

        Returns:
            Color difference score (lower is better)
        """
        # Find the shared edge between pieces
        edge1, edge2 = self._find_shared_edge(piece1, piece2)
        if edge1 is None or edge2 is None:
            return np.float64("inf")  # No shared edge found

        # Extract color values along the edges
        colors1 = self._extract_edge_colors(piece1, edge1)
        colors2 = self._extract_edge_colors(piece2, edge2)

        # Calculate mean squared error between edge colors
        mse = np.mean((colors1 - colors2) ** 2)

        # Normalize the score to be between 0 and 1
        return np.float64(mse / 255.0)

    def _find_shared_edge(
        self, piece1: PuzzlePiece, piece2: PuzzlePiece
    ) -> Tuple[
        Optional[List[Tuple[float, float]]], Optional[List[Tuple[float, float]]]
    ]:
        """Find the shared edge between two pieces.

        Args:
            piece1: First piece
            piece2: Second piece

        Returns:
            Tuple of (edge1, edge2) where each edge is a list of points
            Returns (None, None) if no shared edge is found
        """
        # Convert corners to numpy array for easier manipulation
        corners1 = np.array(piece1.corners)
        corners2 = np.array(piece2.corners)

        # Find the closest points between pieces
        min_dist = float("inf")
        shared_edge1 = None
        shared_edge2 = None

        # Check each edge of piece1 against each edge of piece2
        for i in range(len(corners1)):
            edge1_start = corners1[i]
            edge1_end = corners1[(i + 1) % len(corners1)]

            for j in range(len(corners2)):
                edge2_start = corners2[j]
                edge2_end = corners2[(j + 1) % len(corners2)]

                # Calculate distance between edge midpoints
                mid1 = (edge1_start + edge1_end) / 2
                mid2 = (edge2_start + edge2_end) / 2
                dist = np.linalg.norm(mid1 - mid2)

                if dist < min_dist:
                    min_dist = dist
                    shared_edge1 = [edge1_start.tolist(), edge1_end.tolist()]
                    shared_edge2 = [edge2_start.tolist(), edge2_end.tolist()]

        # If edges are too far apart, they don't share an edge
        if min_dist > 20:  # Threshold for considering edges as shared
            return None, None

        return shared_edge1, shared_edge2

    def _extract_edge_colors(
        self, piece: PuzzlePiece, edge: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Extract color values along an edge of a piece.

        Args:
            piece: The puzzle piece
            edge: List of two points defining the edge

        Returns:
            Array of color values along the edge
        """
        # Convert edge points to integers
        start = np.array(edge[0], dtype=np.int32)
        end = np.array(edge[1], dtype=np.int32)

        # Calculate number of points to sample along the edge
        length = int(np.linalg.norm(end - start))
        num_points = max(10, length // 2)  # At least 10 points, or half the length

        # Generate points along the edge
        points = np.linspace(start, end, num_points, dtype=np.int32)

        # Extract colors at each point
        colors = []
        for point in points:
            if (
                0 <= point[0] < piece.image.shape[1]
                and 0 <= point[1] < piece.image.shape[0]
            ):
                colors.append(piece.image[point[1], point[0]])

        return np.array(colors)

    def _has_straight_edge(self, piece: PuzzlePiece, edge_type: str) -> bool:
        """Check if a piece has a straight edge on the specified side.

        Args:
            piece: Piece to check
            edge_type: Type of edge to check for ("top", "bottom", "left", "right")

        Returns:
            True if piece has a straight edge on the specified side
        """
        # Find the edge on the specified side
        edge = self._find_edge_on_side(piece, edge_type)
        if edge is None:
            return False

        # Calculate how straight the edge is
        return self._is_edge_straight(edge)

    def _find_edge_on_side(
        self, piece: PuzzlePiece, edge_type: str
    ) -> Optional[List[Tuple[float, float]]]:
        """Find the edge of a piece that's on the specified side.

        Args:
            piece: The puzzle piece
            edge_type: Type of edge to find ("top", "bottom", "left", "right")

        Returns:
            List of two points defining the edge, or None if no edge found
        """
        corners = np.array(piece.corners)

        # Define the expected direction for each edge type
        directions = {
            "top": (0, -1),  # Up
            "bottom": (0, 1),  # Down
            "left": (-1, 0),  # Left
            "right": (1, 0),  # Right
        }

        if edge_type not in directions:
            return None

        expected_dir = np.array(directions[edge_type])
        best_edge = None
        best_alignment = -1

        # Check each edge
        for i in range(len(corners)):
            edge_start = corners[i]
            edge_end = corners[(i + 1) % len(corners)]

            # Calculate edge direction
            edge_dir = edge_end - edge_start
            edge_dir = edge_dir / np.linalg.norm(edge_dir)

            # Calculate alignment with expected direction
            alignment = np.abs(np.dot(edge_dir, expected_dir))

            if alignment > best_alignment:
                best_alignment = alignment
                best_edge = [edge_start.tolist(), edge_end.tolist()]

        # If the best alignment is too low, no edge found
        if best_alignment < 0.7:  # Threshold for considering an edge as straight
            return None

        return best_edge

    def _is_edge_straight(self, edge: List[Tuple[float, float]]) -> bool:
        """Check if an edge is approximately straight.

        Args:
            edge: List of two points defining the edge

        Returns:
            True if the edge is approximately straight
        """
        # Convert to numpy array
        points = np.array(edge)

        # Calculate the angle of the edge
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

        # Check if the angle is close to 0, 90, 180, or 270 degrees
        return any(abs(angle - target) < 15 for target in [0, 90, 180, 270])

    def _fill_interior(self) -> None:
        """Fill the interior of the puzzle by matching pieces."""
        rows, cols = self.solution_matrix.shape

        # Fill remaining positions
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if self.solution_matrix[row, col] == -1:
                    self._find_and_place_interior_piece(row, col)

    def _find_and_place_interior_piece(self, row: int, col: int) -> None:
        """Find and place an interior piece at the specified position.

        Args:
            row: Row position
            col: Column position
        """
        adjacent_pieces = self._get_adjacent_pieces(row, col)

        # Find best matching piece
        best_piece = None
        best_score = float("-inf")

        for piece in self.pieces:
            if piece.id in self.solution_matrix:
                continue

            score = self._calculate_interior_match_score(piece, adjacent_pieces)
            if score > best_score:
                best_score = score
                best_piece = piece

        if best_piece is not None:
            self.solution_matrix[row, col] = best_piece.id

    def _calculate_interior_match_score(
        self, piece: PuzzlePiece, adjacent_pieces: List[Optional[PuzzlePiece]]
    ) -> np.float64:
        """Calculate how well a piece matches with its adjacent pieces in the interior.

        Args:
            piece: Piece to evaluate
            adjacent_pieces: List of adjacent pieces

        Returns:
            Match score (higher is better)
        """
        score = 0.0

        # Check color matching with all adjacent pieces
        for adj_piece in adjacent_pieces:
            if adj_piece is not None:
                color_diff = self._calculate_edge_color_difference(piece, adj_piece)
                score -= color_diff

        return np.float64(score)

    def _visualize_solution(self) -> None:
        """Create a visualization of the current puzzle solution state."""
        if not DEBUG:
            return

        # Create a blank image for the solution
        rows, cols = self.solution_matrix.shape
        piece_size = 100  # Size of each piece in the visualization
        solution_image = np.zeros(
            (rows * piece_size, cols * piece_size, 3), dtype=np.uint8
        )

        # Draw each piece in its position
        for row in range(rows):
            for col in range(cols):
                piece_id = self.solution_matrix[row, col]
                if piece_id != -1:
                    piece = next(p for p in self.pieces if p.id == piece_id)

                    # Calculate position in solution image
                    x = col * piece_size
                    y = row * piece_size

                    # Draw piece outline
                    corners = np.array(piece.corners, dtype=np.int32)
                    # Scale and translate corners to fit in the piece's position
                    corners = corners * (piece_size / max(piece.image.shape[:2]))
                    corners = corners + np.array([x, y])
                    cv2.polylines(solution_image, [corners], True, (0, 255, 0), 2)

                    # Draw piece ID
                    cv2.putText(
                        solution_image,
                        str(piece_id),
                        (x + piece_size // 2, y + piece_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

        self._save_debug_image("current_solution", solution_image)
