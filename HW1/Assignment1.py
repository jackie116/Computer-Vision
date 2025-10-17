import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # def check_package_versions(self):
    #     # Ungraded
    #     import numpy as np
    #     import matplotlib
    #     import cv2

    #     print(np.__version__)
    #     print(matplotlib.__version__)
    #     print(cv2.__version__)

    def load_and_analyze_image(self):
        Image_data_type = type(self.image)
        Pixel_data_type = self.image.dtype
        Image_shape = self.image.shape

        print(f"Image data type: {Image_data_type}")
        print(f"Pixel data type: {Pixel_data_type}")
        print(f"Image dimensions: {Image_shape}")

        return Image_data_type, Pixel_data_type, Image_shape

    def create_red_image(self):
        red_image = self.image.copy()
        red_image[:, :, 0] = 0
        red_image[:, :, 1] = 0
        return red_image

    def create_photographic_negative(self):
        negative_image = 255 - self.image.copy()
        return negative_image

    def swap_color_channels(self):
        swapped_image = self.image.copy()
        swapped_image = swapped_image[:, :, [2, 1, 0]]
        return swapped_image

    def foliage_detection(self):
        foliage_image = np.zeros(self.image.shape[:2])

        blue_channel = self.image[:, :, 0]
        green_channel = self.image[:, :, 1]
        red_channel = self.image[:, :, 2]

        threshold = (blue_channel < 50) & (green_channel >= 50) & (red_channel < 50)
        foliage_image[threshold] = 255

        return foliage_image

    def shift_image(self):
        shifted_image = np.zeros_like(self.image)
        x, y = shifted_image.shape[:2]
        shifted_image[100:,200:] = self.image[:x-100,:y-200]

        return shifted_image

    def rotate_image(self):
        rotated_image = self.image.copy()
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image

    def similarity_transform(self, scale, theta, shift):
        height, width = self.image.shape[:2]

        transformed_image = np.zeros_like(self.image)

        for y in range(height):
            for x in range(width):
                # translation
                origin_x = x - shift[0]
                origin_y = y - shift[1]

                # rotation
                rotated_x = origin_x * np.cos(np.radians(-theta)) - origin_y * np.sin(np.radians(-theta))
                rotated_y = origin_x * np.sin(np.radians(-theta)) + origin_y * np.cos(np.radians(-theta))

                # scaling
                scaled_x = rotated_x / scale
                scaled_y = rotated_y / scale

                if 0 <= int(round(scaled_y)) < height and 0 <= int(round(scaled_x)) < width:
                    transformed_image[y, x] = self.image[int(round(scaled_y)), int(round(scaled_x))]


        # corners = np.array([
        #     [0, 0],
        #     [width, 0],
        #     [0, height],
        #     [width, height]
        # ], dtype=np.float32)

        # matrix = np.array([
        #     [scale * np.cos(np.radians(theta)), -scale * np.sin(np.radians(theta)), shift[0]],
        #     [scale * np.sin(np.radians(theta)), scale * np.cos(np.radians(theta)), shift[1]]
        # ], dtype=np.float32)

        # transformed_corners = cv2.transform(corners.reshape(1, 4, 2), matrix).reshape(4, 2)

        # min_x = int(np.floor(np.min(transformed_corners[:, 0])))
        # max_x = int(np.ceil(np.max(transformed_corners[:, 0])))
        # min_y = int(np.floor(np.min(transformed_corners[:, 1])))
        # max_y = int(np.ceil(np.max(transformed_corners[:, 1])))
                            
        # new_width = max_x - min_x
        # new_height = max_y - min_y

        # transformed_image = np.zeros((new_height, new_width, 3), dtype=self.image.dtype)
            
        # for y in range(new_height):
        #     for x in range(new_width):
        #         x_new = x + min_x
        #         y_new = y + min_y

        #         # translation
        #         origin_x = x_new - shift[0]
        #         origin_y = y_new - shift[1]

        #         # rotation
        #         rotated_x = origin_x * np.cos(np.radians(-theta)) - origin_y * np.sin(np.radians(-theta))
        #         rotated_y = origin_x * np.sin(np.radians(-theta)) + origin_y * np.cos(np.radians(-theta))

        #         # scaling
        #         scaled_x = rotated_x / scale
        #         scaled_y = rotated_y / scale

        #         if 0 <= int(round(scaled_y)) < height and 0 <= int(round(scaled_x)) < width:
        #             transformed_image[y, x] = self.image[int(round(scaled_y)), int(round(scaled_x))]

        return transformed_image

    def convert_to_grayscale(self):
        gray_image = np.zeros(self.image.shape[:2])
        blue_channel = self.image[:, :, 0].astype(np.float32)
        green_channel = self.image[:, :, 1].astype(np.float32)
        red_channel = self.image[:, :, 2].astype(np.float32)
        gray_image = ((blue_channel * 1 + green_channel * 6 + red_channel* 3) / 10).round().astype(self.image.dtype)
        return gray_image

    def compute_moments(self):
        _, binary_image = cv2.threshold(self.binary_image, 0, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(binary_image)
        m00 = moments['m00']
        m10 = moments['m10']
        m01 = moments['m01']
        x_bar = m10 / m00 if m00 != 0 else 0
        y_bar = m01 / m00 if m00 != 0 else 0
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11']

        # Print the results
        print("First-Order Moments:")
        print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        print("Centralized Moments:")
        print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        print("Second-Order Centralized Moments:")
        print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")

        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11

    def compute_orientation_and_eccentricity(self):
        m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11 = self.compute_moments()
        theta_radian = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        orientation = np.degrees(theta_radian)

        A = mu20 + mu02
        B = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
        lambda_max = (A + B) / 2
        lambda_min = (A - B) / 2

        major_axis_length = 2 * np.sqrt(lambda_max / m00) 
        minor_axis_length = 2 * np.sqrt(lambda_min / m00)

        eccentricity = np.sqrt((lambda_max - lambda_min) / lambda_max)
        # cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color [, thickness[, lineType[, shift]]])
        # Parameters:
        # image: It is the image on which ellipse is to be drawn.
        # centerCoordinates: It is the center coordinates of ellipse. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
        # axesLength: It contains tuple of two variables containing major and minor axis of ellipse (major axis length, minor axis length).
        # angle: Ellipse rotation angle in degrees.
        # startAngle: Starting angle of the elliptic arc in degrees.
        # endAngle: Ending angle of the elliptic arc in degrees.
        # color: It is the color of border line of shape to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
        # thickness: It is the thickness of the shape border line in px. Thickness of -1 px will fill the shape by the specified color.
        # lineType: This is an optional parameter.It gives the type of the ellipse boundary.
        # shift: This is an optional parameter. It denotes the number of fractional bits in the coordinates of the center and values of axes.

        # Return Value: It returns an image.
        canvas = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
        print(f"x_bar: {x_bar}, y_bar: {y_bar}")
        print(f"major_axis_length: {major_axis_length}, minor_axis_length: {minor_axis_length}")
        glasses_with_ellipse = cv2.ellipse(canvas,    
                                           (int(round(x_bar)), int(round(y_bar))), 
                                           (int(round(major_axis_length / 2)), int(round(minor_axis_length / 2))), 
                                           -orientation, 
                                           0, 
                                           360, 
                                           (0, 0, 255), 
                                           1)
        return orientation, eccentricity, glasses_with_ellipse


if __name__ == "__main__":

    assignment = ComputerVisionAssignment("original_image.png", "binary_image.png")

    # Task 0: Check package versions
    # assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # # Task 2: Create a red image
    red_image = assignment.create_red_image()
    # cv2.imwrite("red_image.png", red_image)

    # # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()
    # cv2.imwrite("negative_image.png", negative_image)    

    # # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()
    # cv2.imwrite("swapped_image.png", swapped_image)

    # # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()
    # cv2.imwrite("foliage_image.png", foliage_image)

    # # Task 6: Shift the image
    shifted_image = assignment.shift_image()
    # cv2.imwrite("shifted_image.png", shifted_image)

    # # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()
    # cv2.imwrite("rotated_image.png", rotated_image)
    
    # # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )
    # cv2.imwrite("transformed_image.png", transformed_image)

    # # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()
    # cv2.imwrite("gray_image.png", gray_image)

    glasses_assignment = ComputerVisionAssignment(
        "glasses_outline.png", "glasses_outline.png"
    )

    # # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )
    # cv2.imwrite("glasses_with_ellipse.png", glasses_with_ellipse)
