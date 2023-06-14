import segno
from PIL import Image, ImageDraw, ImageOps
from typing import Union, List
import random
import numpy as np
import cv2
from itertools import product


class QRCodeGenerator:
    def __init__(
        self,
        input_string,
        version=4,
        pixel_size=16,
        padding=64,
        image_size=512,
        bg_color=(128, 128, 128),
        quiet_color=(225, 225, 225),
        module_color=(50, 50, 50),
        pattern_color=(0, 0, 0),
        transform_amount=2,
        code_scale=0.85,
    ):
        self.input_string = input_string
        self.version = version
        self.pixel_size = pixel_size
        self.padding = padding
        self.image_size = image_size
        self.bg_color = bg_color
        self.quiet_color = quiet_color
        self.module_color = module_color
        self.pattern_color = pattern_color
        self.transform_amount = transform_amount
        self.code_scale = code_scale
        self.qrcode = None
        self.matrix = None
        self.pattern_mask = None

    def _is_marker(self, x: int, y: int) -> bool:
        if self.pattern_mask[x][y] == 1:
            return True
        return False

    def _is_alignment_within_finder(self, pos, qr_size) -> bool:
        x, y = pos
        finder_positions = [(0, 0), (0, qr_size - 7), (qr_size - 7, 0)]
        alignment_size = 5

        for fp_x, fp_y in finder_positions:
            if any(
                (x + i, y + j)
                in [(fp_x + dx, fp_y + dy) for dx in range(7) for dy in range(7)]
                for i in range(alignment_size)
                for j in range(alignment_size)
            ):
                return False
        return True

    def _get_alignment_positions(self):
        positions = []
        if self.version > 1:
            n_patterns = self.version // 7 + 2
            first_pos = 6
            positions.append(first_pos)
            matrix_width = 17 + 4 * self.version
            last_pos = matrix_width - 1 - first_pos
            second_last_pos = (
                (first_pos + last_pos * (n_patterns - 2) + (n_patterns - 1) // 2)
                // (n_patterns - 1)
            ) & -2
            pos_step = last_pos - second_last_pos
            second_pos = last_pos - (n_patterns - 2) * pos_step
            positions.extend(range(second_pos, last_pos + 1, pos_step))
        positions = list(product(positions, repeat=2))
        positions_clean = []
        for position in positions:
            if self._is_alignment_within_finder(position, matrix_width):
                positions_clean.append(position)
        return positions_clean

    def _add_finder_pattern(self, mask, x, y):
        for i in range(7):
            mask[x + i][y] = 1
            mask[x + i][y + 6] = 1
            mask[x][y + i] = 1
            mask[x + 6][y + i] = 1

        for i in range(2, 5):
            for j in range(2, 5):
                mask[x + i][y + j] = 1

    def _add_alignment_pattern(self, mask, x, y):
        x = x - 2
        y = y - 2
        alignment_pattern = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

        for i in range(5):
            for j in range(5):
                mask[x + i][y + j] = alignment_pattern[i][j]

    def _generate_qr_code_mask(self) -> List[List[int]]:
        size = self.version * 4 + 17  # Calculate the size of the QR code matrix
        mask = [[0] * size for _ in range(size)]  # Initialize the matrix with all zeros

        # Add the finder patterns
        self._add_finder_pattern(mask, 0, 0)
        self._add_finder_pattern(mask, size - 7, 0)
        self._add_finder_pattern(mask, 0, size - 7)

        # Add the alignment patterns
        if self.version >= 2:
            alignment_coords = self._get_alignment_positions()
            for coord in alignment_coords:
                self._add_alignment_pattern(mask, coord[0], coord[1])

        return mask

    def _get_round_corners(self, x: int, y: int) -> Union[bool, bool, bool, bool]:
        rows = len(self.matrix)
        cols = len(self.matrix[0])

        # Determine if the pixel is located at the edge of the matrix
        is_top_edge = y == 0
        is_right_edge = x == cols - 1
        is_bottom_edge = y == rows - 1
        is_left_edge = x == 0

        # Determine if the pixel has neighboring pixels on each side
        has_top_neighbor = not is_top_edge and self.matrix[y - 1][x]
        has_right_neighbor = not is_right_edge and self.matrix[y][x + 1]
        has_bottom_neighbor = not is_bottom_edge and self.matrix[y + 1][x]
        has_left_neighbor = not is_left_edge and self.matrix[y][x - 1]

        # Randomly select one corner to be rounded for each side with no neighbors
        top_left = (
            not has_top_neighbor
            and not has_left_neighbor
            and random.choice([True, False])
        )
        top_right = (
            not has_top_neighbor
            and not has_right_neighbor
            and random.choice([True, False])
        )
        bottom_right = (
            not has_bottom_neighbor
            and not has_right_neighbor
            and random.choice([True, False])
        )
        bottom_left = (
            not has_bottom_neighbor
            and not has_left_neighbor
            and random.choice([True, False])
        )

        return [top_left, top_right, bottom_right, bottom_left]

    def _apply_perspective_transform(self, image: Image):
        width, height = image.size

        # Define the source and destination points for the perspective transformation
        source_points = np.float32(
            [
                (self.padding, self.padding),
                (width - self.padding, self.padding),
                (width - self.padding, height - self.padding),
                (self.padding, height - self.padding),
            ]
        )
        destination_points = np.float32(
            [
                (self.padding + self.pixel_size * self.transform_amount, self.padding),
                (
                    width - (self.padding + self.pixel_size * self.transform_amount),
                    self.padding + self.pixel_size * self.transform_amount,
                ),
                (
                    width - self.padding,
                    height - (self.padding + self.pixel_size * self.transform_amount),
                ),
                (self.padding, height - self.padding),
            ]
        )

        # Create a perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(
            source_points, destination_points
        )

        # Apply the perspective transformation using OpenCV
        transformed_image = cv2.warpPerspective(
            np.array(image),
            perspective_matrix,
            (width, height),
            borderValue=self.bg_color,
        )

        # Convert the transformed image to PIL format
        transformed_image = Image.fromarray(transformed_image)

        # Return the transformed image
        return transformed_image

    def generate_qr_code(self):
        # Create QR code and binary mask
        self.qrcode = segno.make(self.input_string, error="H", version=self.version)
        self.matrix = self.qrcode.matrix

        # Create image
        size = len(self.matrix) * self.pixel_size + 2 * self.padding
        image = Image.new(mode="RGB", size=(size, size), color=self.quiet_color)
        draw = ImageDraw.Draw(image)

        # Generate finder and alignment pattern mask
        self.pattern_mask = self._generate_qr_code_mask()

        # Draw QR code
        for y, row in enumerate(self.matrix):
            for x, c in enumerate(row):
                if c == 1:
                    color = (
                        self.pattern_color
                        if self._is_marker(x, y)
                        else self.module_color
                    )
                    draw.rounded_rectangle(
                        (
                            (
                                self.padding + x * self.pixel_size,
                                self.padding + y * self.pixel_size,
                            ),
                            (
                                self.padding + x * self.pixel_size + self.pixel_size,
                                self.padding + y * self.pixel_size + self.pixel_size,
                            ),
                        ),
                        radius=7,
                        fill=color,
                        corners=self._get_round_corners(x, y),
                    )

        # Scale down the image
        scaled_down_image = ImageOps.scale(
            image, (self.image_size / image.width) * self.code_scale
        )
        final_image = Image.new(
            "RGB", (self.image_size, self.image_size), color=self.bg_color
        )
        x = (final_image.width - scaled_down_image.width) // 2
        y = (final_image.height - scaled_down_image.height) // 2
        final_image.paste(scaled_down_image, (x, y))

        # Apply perspective transform
        final_image = self._apply_perspective_transform(image=final_image)

        return final_image

    def save_qr_code(self, filename):
        qr_code_image = self.generate_qr_code()
        qr_code_image.save(filename)
