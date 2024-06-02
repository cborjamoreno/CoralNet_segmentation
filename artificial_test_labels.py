from PIL import Image
import os

# Directory containing the test images
test_images_dir = 'datasets/Sebens_MA_LTM/images/test'

# Directory to save the dummy labels
test_labels_dir = 'datasets/Sebens_MA_LTM/labels/test'

# Get the list of test images
test_images = os.listdir(test_images_dir)

# Create a black image for each test image
for image_name in test_images:
    # Open the test image
    image = Image.open(os.path.join(test_images_dir, image_name))

    # Create a black image of the same size
    black_image = Image.new('L', image.size)

    # Save the black image in the test labels directory
    black_image.save(os.path.join(test_labels_dir, image_name))