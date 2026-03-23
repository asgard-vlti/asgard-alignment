
def raster_scan_with_orientation(starting_point, dx, dy, width, height, orientation=0):
    """
    Generates a raster scan pattern within a defined rectangular area and rotates it by a given orientation.

    Parameters:
    starting_point (tuple): The initial (x, y) point to start the raster scan.
    dx (float): Step size in the x-direction.
    dy (float): Step size in the y-direction.
    width (float): Total width of the scan area.
    height (float): Total height of the scan area.
    orientation (float): Orientation angle in degrees (rotation counterclockwise).

    Returns:
    list: A list of tuples where each tuple contains (x, y) positions for the scan.
    """
    x_start, y_start = starting_point
    scan_points = []

    # Define the bounds of the scan area
    x_min = 0
    x_max = width
    y_min = 0
    y_max = height

    # Initialize y and direction for x movement
    y = y_min
    direction = 1  # 1 for left-to-right, -1 for right-to-left

    while y <= y_max:
        # Generate a row of points
        row_points = []
        if direction == 1:  # Left-to-right
            x = x_min
            while x <= x_max:
                row_points.append((x, y))
                x += dx
        else:  # Right-to-left
            x = x_max
            while x >= x_min:
                row_points.append((x, y))
                x -= dx

        # Add the row to the scan points
        scan_points.extend(row_points)

        # Move to the next row and flip direction
        y += dy
        direction *= -1

    # Rotate points based on the orientation angle
    angle_rad = np.radians(orientation)
    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

    # Apply rotation and translate back to the starting point
    rotated_points = []
    for x, y in scan_points:
        x_rot = cos_theta * x - sin_theta * y
        y_rot = sin_theta * x + cos_theta * y
        rotated_points.append((x_rot + x_start, y_rot + y_start))

    return rotated_points