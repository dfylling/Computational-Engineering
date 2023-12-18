
def upper_concave(points):
    """Compute the upper concave envelope of a set of points."""
    # First, sort the points by x-coordinate
    points = sorted(points, key=lambda x: x[0])

    # Initialize the stack with the first two points
    stack = [points[0], points[1]]

    for i in range(2, len(points)):
        while len(stack) > 1:
            # Compute the slope of the line between the current point and the top of the stack
            slope = (points[i][1] - stack[-1][1]) / (points[i][0] - stack[-1][0])

            # If the slope is negative, then the current point is below the line defined by the top two points of the stack
            if slope < (stack[-1][1] - stack[-2][1]) / (stack[-1][0] - stack[-2][0]):
                break

            # If the slope is positive, then pop the top point from the stack
            stack.pop()

        # Add the current point to the stack
        stack.append(points[i])

    return stack

def lower_convex(points):
    """Compute the lower convex envelope of a set of points."""
    # First, sort the points by x-coordinate
    points = sorted(points, key=lambda x: x[0])

    # Initialize the stack with the first two points
    stack = [points[0], points[1]]

    for i in range(2, len(points)):
        while len(stack) > 1:
            # Compute the slope of the line between the current point and the top of the stack
            slope = (points[i][1] - stack[-1][1]) / (points[i][0] - stack[-1][0])

            # If the slope is positive, then the current point is below the line defined by the top two points of the stack
            if slope > (stack[-1][1] - stack[-2][1]) / (stack[-1][0] - stack[-2][0]):
                break

            # If the slope is negative, then pop the top point from the stack
            stack.pop()

        # Add the current point to the stack
        stack.append(points[i])
    return stack