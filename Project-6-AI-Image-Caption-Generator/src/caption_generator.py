from PIL import Image
import os


def generate_caption(image_path):
    """
    Offline demo caption generator.

    This version does not download or use a large
    deep-learning model.
    """

    if not os.path.exists(image_path):
        return "Image file not found."

    try:
        image = Image.open(image_path)
        width, height = image.size

        if width > height:
            caption = (
                "A landscape-oriented image with "
                "visual content in the scene."
            )
        elif height > width:
            caption = (
                "A portrait-oriented image showing "
                "visual content in the scene."
            )
        else:
            caption = (
                "A square image containing visual "
                "content."
            )

        return caption

    except Exception as error:
        return f"Could not process image: {error}"


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    caption = generate_caption(image_path)

    print()
    print("Generated Caption:")
    print(caption)