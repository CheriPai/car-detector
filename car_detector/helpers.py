import imutils


def pyramid(image, scale=1.5, min_size=(30, 30)):
    """Creates an image pyramid
       yields an image for each iteration
    """
    yield image
    while image.shape[0] >= min_size[1] and image.shape[1] >= min_size[0]:
        width = int(image.shape[1] / scale)
        image = imutils.resize(image, width=width)
        yield image


def sliding_window(image, step_size, window_size):
    """Slides a window of window_size across image
       yielding an image for each iteration
    """
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield(x, y, image[y:y+window_size[1], x:x+window_size[0]])
