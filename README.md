## Milestone 2 <br>
We observed inconsistencies in file extensions across the images, with some using uppercase letters (e.g., "JPG" instead of "jpg") and others varying in format (png, jpeg, jpg), so we need to standardize all file extensions to lowercase. The dataset contains 400 images across four classes: Bus, Car, Motorcycle, and Truck, with unique image dimensions per class as follows: Bus – 82, Car – 71, Motorcycle – 74, and Truck – 78. For preprocessing, we will first resize all images to a consistent dimension, ensuring uniformity across the dataset. The target dimensions will be decided based on our approach, whether using a pre-trained neural network or developing a custom model. If resizing alters the desired aspect ratios, we will apply zero padding to center the resized images on a blank canvas, preserving their original proportions. Additionally, we will implement data augmentation techniques to enhance the dataset’s variability, such as rotation, flipping, and scaling, which will help improve the robustness of our model. These preprocessing and augmentation steps will ensure a well-structured, versatile dataset ready for analysis and model training. <br>
## Milestone 3 <br>
*Note*: You may have to refresh the page if the ``milestone_3.ipynb`` notebook can't render (it'll load eventually). The notebook contains all major and finalized preprocessing, feature extraction, and our baseline model. <br>
1: Finish Major Preprocessing

Finish major preprocessing, this includes scaling and/or transforming your data, imputing your data, encoding your data, feature expansion, Feature expansion (example is taking features and generating new features by transforming via polynomial, log multiplication of features).

2: Train your first model

3: Evaluate your model and compare training vs. test error

4: Answer the questions: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?

5: Update your README.md to include your new work and updates you have all added. Make sure to upload all code and notebooks. Provide links in your README.md

6. Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

Please make sure preprocessing is complete and your first model has been trained. If you are doing supervised learning include example ground truth and predictions for train, validation and test. 
