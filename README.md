## Milestone 2 <br>
We observed inconsistencies in file extensions across the images, with some using uppercase letters (e.g., "JPG" instead of "jpg") and others varying in format (png, jpeg, jpg), so we need to standardize all file extensions to lowercase. The dataset contains 400 images across four classes: Bus, Car, Motorcycle, and Truck, with unique image dimensions per class as follows: Bus – 82, Car – 71, Motorcycle – 74, and Truck – 78. For preprocessing, we will first resize all images to a consistent dimension, ensuring uniformity across the dataset. The target dimensions will be decided based on our approach, whether using a pre-trained neural network or developing a custom model. If resizing alters the desired aspect ratios, we will apply zero padding to center the resized images on a blank canvas, preserving their original proportions. Additionally, we will implement data augmentation techniques to enhance the dataset’s variability, such as rotation, flipping, and scaling, which will help improve the robustness of our model. These preprocessing and augmentation steps will ensure a well-structured, versatile dataset ready for analysis and model training. <br>
## Milestone 3 <br>
