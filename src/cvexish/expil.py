from PIL import Image, ImageDraw, ImageFont


def concat_image_on_grid(images: list[Image.Image], columns: int) -> Image.Image:
    """Concatenates multiple images into a grid.

    Args:
        images (list[Image.Image]): List of images to concatenate.
        columns (int): Number of columns in the grid.
    Result:
        Image: The concatenated grid
    """
    assert all(images[0].size == i.size for i in images[1:])

    w, h = images[0].size
    rows = len(images) // columns + 1
    grid = Image.new("RGB", size=(columns * w, rows * h), color="white")
    for i, image in enumerate(images):
        grid.paste(image, box=(i % columns * w, i // columns * h))

    return grid


def resize_and_pad(
    image: Image.Image,
    target_width: int,
    target_height: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Resize an image to fit within target dimensions, preserving aspect ratio.
    If the image is smaller than the target size, it is padded with the specified color.
    If the image is larger, it is shrunk while retaining aspect ratio.

    Args:
        image (PIL.Image.Image): Input image.
        target_width (int): Target width.
        target_height (int): Target height.
        color (tuple): RGB color for padding. Default is white (255, 255, 255).

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    # Original dimensions
    original_width, original_height = image.size

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Determine new size based on aspect ratio
    if target_aspect_ratio > aspect_ratio:
        # Match height, adjust width
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        # Match width, adjust height
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with the target size and the specified background color
    new_image = Image.new("RGB", (target_width, target_height), color)

    # Calculate padding to center the resized image
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2

    # Paste resized image onto the new image
    new_image.paste(resized_image, (pad_left, pad_top))

    return new_image


def text_image(
    text: str,
    width: int = 300,
    height: int = 200,
    background_color: str = "white",
    text_color: str = "black",
    font_size: int = 40,
    font_name: str | None = "ヒラギノ丸ゴ ProN W4.ttc",
) -> Image.Image:
    """Create an image with text centered in the middle.

    Args:
        text (str): The text to draw.
        width (int): Width of the image.
        height (int): Height of the image.
        background_color (str): Background color of the image.
        text_color (str): Text color.
        font_size (int): Font size.
        font_name (str): Path to the font file. Default is "ヒラギノ丸ゴ ProN W4.ttc".
    Result:
        Image: The generated image.
    """
    image = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(image)

    if font_name is None:
        font = ImageFont.load_default(size=font_size)
    else:
        font = ImageFont.truetype(font_name, size=font_size)

    # Calculate text position
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2

    # Draw the text.
    draw.text((text_x, text_y), text, fill=text_color, font=font, size=font_size)

    return image
