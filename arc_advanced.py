"""
Advanced ARC Solving Techniques
This module contains sophisticated approaches for solving ARC tasks:
- Object detection and manipulation
- Pattern mining and abstraction
- Graph-based reasoning
- DSL (Domain Specific Language) based program synthesis
- Test-time adaptation and augmentation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass
import itertools
from scipy import ndimage
from scipy.spatial.distance import cdist


# ============================================================================
# OBJECT DETECTION AND MANIPULATION
# ============================================================================

@dataclass
class ARCObject:
    """Represents a detected object in an ARC grid."""
    pixels: Set[Tuple[int, int]]  # Set of (row, col) coordinates
    color: int
    bounding_box: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)

    @property
    def height(self):
        return self.bounding_box[2] - self.bounding_box[0] + 1

    @property
    def width(self):
        return self.bounding_box[3] - self.bounding_box[1] + 1

    @property
    def area(self):
        return len(self.pixels)

    def get_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Get binary mask for this object."""
        mask = np.zeros(shape, dtype=bool)
        for r, c in self.pixels:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
        return mask


class ObjectDetector:
    """Detects objects in ARC grids using connected components."""

    def __init__(self, background_color: int = 0):
        self.background_color = background_color

    def detect_objects(self, grid: np.ndarray) -> List[ARCObject]:
        """Detect all objects in the grid."""
        objects = []

        # Find all unique colors (excluding background)
        colors = np.unique(grid)
        colors = colors[colors != self.background_color]

        for color in colors:
            # Create binary mask for this color
            mask = (grid == color)

            # Find connected components
            labeled, num_features = ndimage.label(mask)

            for obj_id in range(1, num_features + 1):
                obj_mask = (labeled == obj_id)
                pixels = set(zip(*np.where(obj_mask)))

                if not pixels:
                    continue

                rows, cols = zip(*pixels)
                bbox = (min(rows), min(cols), max(rows), max(cols))

                objects.append(ARCObject(
                    pixels=pixels,
                    color=color,
                    bounding_box=bbox
                ))

        return objects

    def extract_object_grid(self, grid: np.ndarray, obj: ARCObject) -> np.ndarray:
        """Extract a rectangular grid containing the object."""
        min_r, min_c, max_r, max_c = obj.bounding_box
        return grid[min_r:max_r+1, min_c:max_c+1].copy()


class ObjectTransformer:
    """Applies transformations to objects."""

    @staticmethod
    def translate(obj: ARCObject, dr: int, dc: int) -> ARCObject:
        """Translate object by (dr, dc)."""
        new_pixels = {(r + dr, c + dc) for r, c in obj.pixels}
        min_r, min_c, max_r, max_c = obj.bounding_box
        new_bbox = (min_r + dr, min_c + dc, max_r + dr, max_c + dc)
        return ARCObject(new_pixels, obj.color, new_bbox)

    @staticmethod
    def scale(obj: ARCObject, scale_factor: int) -> ARCObject:
        """Scale object by integer factor."""
        new_pixels = set()
        for r, c in obj.pixels:
            for dr in range(scale_factor):
                for dc in range(scale_factor):
                    new_pixels.add((r * scale_factor + dr, c * scale_factor + dc))

        min_r, min_c, max_r, max_c = obj.bounding_box
        new_bbox = (min_r * scale_factor, min_c * scale_factor,
                   (max_r + 1) * scale_factor - 1, (max_c + 1) * scale_factor - 1)

        return ARCObject(new_pixels, obj.color, new_bbox)

    @staticmethod
    def rotate(obj: ARCObject, k: int = 1) -> ARCObject:
        """Rotate object 90 degrees k times."""
        # Convert to grid, rotate, convert back
        min_r, min_c, max_r, max_c = obj.bounding_box
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        grid = np.zeros((h, w), dtype=int)
        for r, c in obj.pixels:
            grid[r - min_r, c - min_c] = 1

        rotated = np.rot90(grid, k)
        new_pixels = {(r, c) for r, c in zip(*np.where(rotated > 0))}

        new_h, new_w = rotated.shape
        new_bbox = (0, 0, new_h - 1, new_w - 1)

        return ARCObject(new_pixels, obj.color, new_bbox)

    @staticmethod
    def flip(obj: ARCObject, axis: int = 0) -> ARCObject:
        """Flip object along axis (0=horizontal, 1=vertical)."""
        min_r, min_c, max_r, max_c = obj.bounding_box

        if axis == 0:  # Flip horizontally
            new_pixels = {(max_r - r + min_r, c) for r, c in obj.pixels}
        else:  # Flip vertically
            new_pixels = {(r, max_c - c + min_c) for r, c in obj.pixels}

        return ARCObject(new_pixels, obj.color, obj.bounding_box)


class ObjectBasedSolver:
    """Solver that uses object detection and manipulation."""

    def __init__(self):
        self.detector = ObjectDetector()
        self.transformer = ObjectTransformer()

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Attempt to solve using object-based reasoning."""

        # Analyze training examples to find transformation pattern
        transformations = self._analyze_transformations(train_pairs)

        if not transformations:
            return None

        # Apply most common transformation to test input
        return self._apply_transformation(test_input, transformations[0])

    def _analyze_transformations(self, train_pairs):
        """Analyze what transformations are applied to objects."""
        transformations = []

        for inp, out in train_pairs:
            # Detect objects in input and output
            inp_objects = self.detector.detect_objects(inp)
            out_objects = self.detector.detect_objects(out)

            # Try to match objects and infer transformation
            # This is a simplified version - real implementation would be more sophisticated
            if len(inp_objects) == len(out_objects):
                for inp_obj, out_obj in zip(inp_objects, out_objects):
                    if inp_obj.color == out_obj.color:
                        # Check for translation
                        dr = out_obj.bounding_box[0] - inp_obj.bounding_box[0]
                        dc = out_obj.bounding_box[1] - inp_obj.bounding_box[1]

                        if dr != 0 or dc != 0:
                            transformations.append(('translate', (dr, dc)))

        return transformations

    def _apply_transformation(self, grid, transformation):
        """Apply a transformation to the grid."""
        trans_type, params = transformation

        if trans_type == 'translate':
            dr, dc = params
            result = np.zeros_like(grid)
            h, w = grid.shape

            for r in range(h):
                for c in range(w):
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < h and 0 <= new_c < w:
                        result[new_r, new_c] = grid[r, c]

            return result

        return grid.copy()


# ============================================================================
# PATTERN MINING AND ABSTRACTION
# ============================================================================

class PatternMiner:
    """Mines patterns and abstractions from ARC tasks."""

    @staticmethod
    def find_repeating_patterns(grid: np.ndarray) -> List[Tuple[np.ndarray, List[Tuple[int, int]]]]:
        """Find repeating patterns in the grid."""
        patterns = []
        h, w = grid.shape

        # Try different pattern sizes
        for ph in range(1, h // 2 + 1):
            for pw in range(1, w // 2 + 1):
                pattern = grid[:ph, :pw]
                locations = []

                # Search for this pattern
                for r in range(h - ph + 1):
                    for c in range(w - pw + 1):
                        if np.array_equal(grid[r:r+ph, c:c+pw], pattern):
                            locations.append((r, c))

                if len(locations) > 1:
                    patterns.append((pattern, locations))

        return patterns

    @staticmethod
    def detect_symmetry(grid: np.ndarray) -> Dict[str, bool]:
        """Detect various types of symmetry in the grid."""
        symmetries = {
            'horizontal': np.array_equal(grid, np.flip(grid, axis=0)),
            'vertical': np.array_equal(grid, np.flip(grid, axis=1)),
            'diagonal': np.array_equal(grid, grid.T) if grid.shape[0] == grid.shape[1] else False,
            'rotational_90': np.array_equal(grid, np.rot90(grid, k=1)),
            'rotational_180': np.array_equal(grid, np.rot90(grid, k=2)),
        }
        return symmetries

    @staticmethod
    def find_grid_arithmetic(inp: np.ndarray, out: np.ndarray) -> Optional[Callable]:
        """Try to find arithmetic relationship between input and output."""
        if inp.shape != out.shape:
            return None

        # Check for simple arithmetic operations
        diff = out - inp
        if np.all(diff == diff[0, 0]):
            # Constant addition
            constant = diff[0, 0]
            return lambda x: x + constant

        ratio = np.zeros_like(out, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = out / inp
            if np.all(np.isfinite(ratio)) and np.all(ratio == ratio[0, 0]):
                # Constant multiplication
                multiplier = ratio[0, 0]
                return lambda x: x * multiplier

        return None

    @staticmethod
    def detect_tiling(grid: np.ndarray) -> Optional[Tuple[np.ndarray, int, int]]:
        """Detect if grid is a tiling of a smaller pattern."""
        h, w = grid.shape

        # Try different tile sizes
        for th in range(1, h // 2 + 1):
            if h % th != 0:
                continue

            for tw in range(1, w // 2 + 1):
                if w % tw != 0:
                    continue

                tile = grid[:th, :tw]
                is_tiling = True

                # Check if entire grid is this tile repeated
                for r in range(0, h, th):
                    for c in range(0, w, tw):
                        if not np.array_equal(grid[r:r+th, c:c+tw], tile):
                            is_tiling = False
                            break
                    if not is_tiling:
                        break

                if is_tiling:
                    return (tile, h // th, w // tw)

        return None


# ============================================================================
# DSL-BASED PROGRAM SYNTHESIS
# ============================================================================

class ARCOperation:
    """Base class for ARC operations in our DSL."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FlipH(ARCOperation):
    def __init__(self):
        super().__init__("flip_h")

    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=0)


class FlipV(ARCOperation):
    def __init__(self):
        super().__init__("flip_v")

    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=1)


class Rotate90(ARCOperation):
    def __init__(self, k: int = 1):
        super().__init__(f"rotate_{k*90}")
        self.k = k

    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=self.k)


class ReplaceColor(ARCOperation):
    def __init__(self, old_color: int, new_color: int):
        super().__init__(f"replace_{old_color}_with_{new_color}")
        self.old_color = old_color
        self.new_color = new_color

    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        result[grid == self.old_color] = self.new_color
        return result


class FillColor(ARCOperation):
    def __init__(self, color: int):
        super().__init__(f"fill_{color}")
        self.color = color

    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.full_like(grid, self.color)


class KeepColor(ARCOperation):
    def __init__(self, color: int, background: int = 0):
        super().__init__(f"keep_color_{color}")
        self.color = color
        self.background = background

    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = np.full_like(grid, self.background)
        result[grid == self.color] = self.color
        return result


class ARCProgram:
    """Represents a sequence of operations."""

    def __init__(self, operations: List[ARCOperation]):
        self.operations = operations

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for op in self.operations:
            result = op.apply(result)
        return result

    def __repr__(self):
        return " -> ".join(op.name for op in self.operations)


class ProgramSynthesizer:
    """Synthesizes programs that transform inputs to outputs."""

    def __init__(self, max_program_length: int = 3):
        self.max_program_length = max_program_length
        self.operation_templates = [
            FlipH,
            FlipV,
            lambda: Rotate90(1),
            lambda: Rotate90(2),
            lambda: Rotate90(3),
        ]

    def synthesize(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                   max_attempts: int = 1000) -> Optional[ARCProgram]:
        """Synthesize a program that works for all training pairs."""

        # Extend operation templates with color-specific operations
        all_colors = set()
        for inp, out in train_pairs:
            all_colors.update(np.unique(inp))
            all_colors.update(np.unique(out))

        extended_templates = self.operation_templates.copy()

        # Add color replacement operations
        for c1 in all_colors:
            for c2 in all_colors:
                if c1 != c2:
                    extended_templates.append(lambda c1=c1, c2=c2: ReplaceColor(c1, c2))

        # Try programs of increasing length
        for length in range(1, self.max_program_length + 1):
            # Generate programs of this length
            for attempt in range(max_attempts // self.max_program_length):
                # Randomly sample operations
                ops = [np.random.choice([t() for t in extended_templates])
                       for _ in range(length)]

                program = ARCProgram(ops)

                # Test if program works on all training pairs
                if self._test_program(program, train_pairs):
                    return program

        return None

    def _test_program(self, program: ARCProgram,
                     train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Test if program produces correct outputs for all training pairs."""
        try:
            for inp, expected_out in train_pairs:
                actual_out = program.execute(inp)
                if not np.array_equal(actual_out, expected_out):
                    return False
            return True
        except:
            return False


# ============================================================================
# TEST-TIME ADAPTATION
# ============================================================================

class TestTimeAdapter:
    """Adapts predictions at test time using various strategies."""

    @staticmethod
    def apply_output_constraints(prediction: np.ndarray,
                                 train_outputs: List[np.ndarray]) -> np.ndarray:
        """Apply constraints learned from training outputs."""

        # Constraint 1: Output size
        output_sizes = [out.shape for out in train_outputs]
        if len(set(output_sizes)) == 1:
            # All outputs have same size
            target_size = output_sizes[0]
            if prediction.shape != target_size:
                # Resize prediction
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / prediction.shape[0],
                              target_size[1] / prediction.shape[1])
                prediction = zoom(prediction, zoom_factors, order=0)

        # Constraint 2: Color palette
        output_colors = set()
        for out in train_outputs:
            output_colors.update(np.unique(out))

        # Map prediction colors to nearest valid color
        for color in np.unique(prediction):
            if color not in output_colors:
                # Find nearest valid color
                distances = [abs(color - valid_color) for valid_color in output_colors]
                nearest_color = list(output_colors)[np.argmin(distances)]
                prediction[prediction == color] = nearest_color

        return prediction

    @staticmethod
    def majority_voting(predictions: List[np.ndarray]) -> np.ndarray:
        """Combine multiple predictions using majority voting."""
        if not predictions:
            return None

        # Ensure all predictions have same shape
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) != 1:
            # Different shapes - return first prediction
            return predictions[0]

        shape = predictions[0].shape
        result = np.zeros(shape, dtype=int)

        for i in range(shape[0]):
            for j in range(shape[1]):
                values = [p[i, j] for p in predictions]
                result[i, j] = Counter(values).most_common(1)[0][0]

        return result


# ============================================================================
# ADVANCED ENSEMBLE SOLVER
# ============================================================================

class AdvancedARCSolver:
    """Advanced solver combining all techniques."""

    def __init__(self):
        self.object_solver = ObjectBasedSolver()
        self.pattern_miner = PatternMiner()
        self.program_synthesizer = ProgramSynthesizer()
        self.tta = TestTimeAdapter()

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> List[np.ndarray]:
        """Generate multiple predictions using different techniques."""
        predictions = []

        # Technique 1: Object-based reasoning
        try:
            result = self.object_solver.solve(train_pairs, test_input)
            if result is not None:
                predictions.append(('object_based', result))
        except:
            pass

        # Technique 2: Program synthesis
        try:
            program = self.program_synthesizer.synthesize(train_pairs)
            if program is not None:
                result = program.execute(test_input)
                predictions.append(('program_synthesis', result))
        except:
            pass

        # Technique 3: Pattern-based (tiling detection)
        try:
            # Check if test input is a tiling
            tiling_info = self.pattern_miner.detect_tiling(test_input)
            if tiling_info:
                tile, n_h, n_w = tiling_info
                # Apply transformation to tile and repeat
                for inp, out in train_pairs:
                    if inp.shape == tile.shape:
                        # Transform tile
                        result = np.tile(out, (n_h, n_w))
                        predictions.append(('tiling', result))
                        break
        except:
            pass

        # Apply test-time adaptation
        if predictions:
            train_outputs = [out for _, out in train_pairs]

            adapted_predictions = []
            for name, pred in predictions:
                try:
                    adapted = self.tta.apply_output_constraints(pred, train_outputs)
                    adapted_predictions.append(adapted)
                except:
                    adapted_predictions.append(pred)

            # Majority voting if we have multiple predictions
            if len(adapted_predictions) > 1:
                final_pred = self.tta.majority_voting(adapted_predictions)
                return [final_pred]
            else:
                return [adapted_predictions[0]]

        # Fallback: return input
        return [test_input.copy()]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_grid(grid: np.ndarray, title: str = ""):
    """Simple text-based visualization of a grid."""
    print(f"\n{title}")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid:
        print(" ".join(str(cell) for cell in row))
    print("-" * (grid.shape[1] * 2 + 1))


def analyze_task_statistics(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Analyze statistics of a task."""
    stats = {
        'num_examples': len(train_pairs),
        'input_shapes': [inp.shape for inp, _ in train_pairs],
        'output_shapes': [out.shape for _, out in train_pairs],
        'input_colors': [set(np.unique(inp)) for inp, _ in train_pairs],
        'output_colors': [set(np.unique(out)) for _, out in train_pairs],
        'size_preserved': all(inp.shape == out.shape for inp, out in train_pairs),
        'color_palette': set(),
    }

    for inp, out in train_pairs:
        stats['color_palette'].update(np.unique(inp))
        stats['color_palette'].update(np.unique(out))

    return stats


if __name__ == "__main__":
    # Demo usage
    print("Advanced ARC Solver Module")
    print("=" * 80)

    # Create a simple example
    inp1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    out1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    inp2 = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    out2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])

    test_inp = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    train_pairs = [(inp1, out1), (inp2, out2)]

    # Test advanced solver
    solver = AdvancedARCSolver()
    predictions = solver.solve(train_pairs, test_inp)

    print("\nTest Input:")
    visualize_grid(test_inp)

    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        visualize_grid(pred, f"Prediction {i+1}")

    print("\nDone!")
