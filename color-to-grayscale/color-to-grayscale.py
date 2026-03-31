def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """

    greyscale = []
    for row in image:
        greyscale_row = []
        for pixel in row:
            y = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            greyscale_row.append(y)
        greyscale.append(greyscale_row)

    return greyscale