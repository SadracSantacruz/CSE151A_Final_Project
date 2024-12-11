# CSE 151A: Vehicle Type Recognition Project
_Note_: If you want to take a look at our code, click [here](https://github.com/SadracSantacruz/CSE151A_Final_Project/blob/main/cse151afinalproj.ipynb). You may have to refresh a couple times for it to load properly.
## 1. **Introduction** :car:
The goal of our project was to develop a machine learning system capable of recognizing vehicle types (Car, Motorcycle, Bus, and Truck) based on images. This problem was chosen due to its practical applications in areas such as traffic management, road infrastructure, and safer, smarter, efficient cities.

A robust, high-performing predictive model in this domain could revolutionize how vehicles are monitored and classified, leading to smarter infrastructure planning and improved road safety. Such models could be integrated into intelligent traffic cameras for the following use cases:
- **Traffic Trends**: Classifying vehicles in real-time allows authorities to gather data on the types of vehicles using specific roads at different times (buses during rush hour, trucks at night, etc.). It also can identify what cars exist in different hours of traffic, high-usage vehicle types at a certain hour, etc., which can be used to develop better traffic management strategies.
- **Road Layouts**: Vehicle classification could be used to retrieve what roads are frequently used by certain vehicles. An example of this would be heavy trucks on certain roads, calling for stronger pavement or wider lanes. This type of classification can also be used to optimize traffic lights (i.e. what kind of vehicles are passing by a certain light?)
- **Public Transport System Design**: Classification can help with what routes are car/truck/motorcycle-dominated, and what routes have a low utilization of buses. This can help guide where public transit systems should be expanded, adjusted, or even implemented. It can also help estimate the reliance on personal vehicles versus public transport, being able to identify areas where transit investments can reduce traffic congestion and emissions.
<br>
This project was also a fun execution of our interests as a group, as we all particularly love cars.

<a name="figures"></a>
## 2. **Figures** :chart_with_upwards_trend:

1. **Class Distribution**: Here is a distribution showing the number of images per vehicle class. <br>
   ![image](https://github.com/user-attachments/assets/8b53ae35-e406-402b-b258-06f03cfd93e6)
2. **Sample Images**: Here are some of our (raw) image data within the dataset. <br>
   ![image](https://github.com/user-attachments/assets/48084990-88d1-45c2-a4d1-33cfd2044cf8)
3. **Data Augmentation Examples**: Here are some prepreocessed data of scaled, grayscale, rotated, and flipped images. <br>
   **SCALED**
   ![image](https://github.com/user-attachments/assets/ebca0f51-858b-461a-847d-2e731d7bf0d5)
   **GRAYSCALE**
   ![image](https://github.com/user-attachments/assets/526e3521-3acc-478b-b3ed-9d2cd09d7e63)
   **ROTATE (various degrees)**
   ![image](https://github.com/user-attachments/assets/7f2193d9-edc6-4148-99b3-6c3537e67765)
   **HORIZONTALLY FLIPPED**
   ![image](https://github.com/user-attachments/assets/6e063efc-208f-4945-8ea6-7c41f2f6d53e)
   <br>
   <br>
**Note that there will be additional figures throughout the write-up to show training vs. testing errors, evaluation metrics, confusion matrices, etc.**

## 3. **Methods** :scientist:

### 3.1 **Data Exploration**
- Dataset: Vehicle Type Recognition Dataset from Kaggle
- Exploratory Steps:
  - Visualized class distribution (evenly distributed, see [Figures](#figures))
  - Analyzed image dimensions for standardization needs (see code and output below)
   ```python
   image_paths = []
   folder_path = 'Dataset'
   output_dct = {}
   
   for vehicle in ['Bus', 'Car', 'motorcycle', 'Truck']:
       vehicle_folder = os.path.join(folder_path, vehicle)
       image_dimensions = []
   
       for img in os.listdir(vehicle_folder):
           img_path = os.path.join(vehicle_folder, img)
   
           with Image.open(img_path) as img:
               width, height = img.size
               image_dimensions.append((width, height))
   
       output_dct[vehicle] = image_dimensions
   
   dimension_df = pd.DataFrame(output_dct)
   
   for col in dimension_df.columns:
       num_unique = dimension_df[col].nunique()
       print(f'Number of unique image dimensions for {col}: {num_unique}')
   ```
   ```
   Number of unique image dimensions for Bus: 82
   Number of unique image dimensions for Car: 71
   Number of unique image dimensions for motorcycle: 74
   Number of unique image dimensions for Truck: 78
   ```
  - Visualized example images for each class (see [Figures](#figures))

<a name="embeddings"></a>
### 3.2 **Preprocessing & Data Augmentation**
**Note**: You can see these preprocessed image examples in [Figures](#figures) Section
- **Image Scaling**: Resized all images to 224x224 pixels.
- **Grayscaling**: Converted images to grayscale.
- **Rotation**: Applied rotations of 15°, 30°, 45°, 60°, and 75°.
- **Flipping**: Horizontally flipped images.
- **Feature Extraction**: Used ResNet 50 to generate embeddings (to feed into models that require numerical representation!). See how we generated these embeddings (code below):
  ```python
   import torch
   from torchvision import transforms
   from torchvision.models import resnet50, ResNet50_Weights
   from tqdm import tqdm
   
   resnet50model = resnet50(weights=ResNet50_Weights.DEFAULT)
   resnet50model = torch.nn.Sequential(*list(resnet50model.children())[:-1])
   resnet50model.eval()
   
   normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
   
   transform = transforms.Compose([
       # transforms.Resize((224, 224)),
       transforms.Lambda(lambda img: img.convert("RGB")),
       transforms.ToTensor(),
       normalize
   ])
   
   def preprocess_tensor(img):
       return transform(img)
   
   def extract_embeddings(images, model):
       embeddings = []
       with torch.no_grad():
           for img in tqdm(images, desc="Extracting embeddings"):
               tensor_img = preprocess_tensor(img).unsqueeze(0)
               embedding = model(tensor_img).squeeze().numpy()
               embeddings.append(embedding)
       return embeddings
   
   all_embeddings = extract_embeddings(all_images, resnet50model)
  ```

Preprocessing ensures uniformity in image size, format, and diversity, enabling the model to focus on relevant features for vehicle classification. Augmenting our data also increases data diversity, allowing models to generalize better to varying real-world scenarios (different angles, lighting, orientation). These steps are crucial to enhance the model's vehicle classification performance, ensuring robustness and improved accuracy.

### 3.3 **Model 1: Multi-Class SVM**
Justification: We chose this model due to its effectiveness in handling high-dimensional data (in this case, our ResNet 50 embeddings) and its ability to create clear decision boundaries for classification tasks. By leveraging a linear kernel and the One-vs-Rest approach, it provided a robust baseline for separating vehicle types based on the extracted feature embeddings. Its computational efficiency during training and prediction made it a suitable choice for this problem.
- Hyperparameters Before Tuning:
  - Kernel: Linear
  - Decision Function: One-vs-Rest (OVR)
  - Regularization (C): Not specified (default is set to `C=1`)
- Hyperparameters After Tuning (Grid Search):
  - Kernel: Linear
  - Decision Function: One-vs-Rest (OVR)
  - Regularization (C): 0.1
  ```python
   from sklearn.model_selection import GridSearchCV
   # testing different hyperparameters
   param_grid = {
       'C': [0.1, 1, 10, 100],
       'kernel': ['linear', 'rbf'],
       'decision_function_shape': ['ovr']
   }
   
   # grid search cross-validation to find best hyperparameters for SVM
   grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', return_train_score=True)
   grid_search.fit(X_train_scaled, y_train)
   
   best_params = grid_search.best_params_
  ```
Why These Parameters: We decided to use these tuned hyperparameters for our SVM after Grid Search Cross-Validation as it fetched the best performance out of all combinations. A linear kernel was chosen because the ResNet 50 embeddings are high-dimensional, and a linear boundary is often effective for such data. The One-vs-Rest approach allows the SVM to handle multi-class classification by training a separate binary classifier for each class. A lower C value of 0.1 was selected to prevent overfitting by allowing a softer margin and better generalization.
<br>
<br>
Take a look at how other combinations of hyperparameters performed, and see any signs of underfitting/overfitting:
![image](https://github.com/user-attachments/assets/7f81c088-1c1c-4973-a4d2-785c4bd8ff55)


### 3.4 **Model 2: K-Nearest Neighbor (KNN)**
Justification: We chose this model for its simplicity and effectiveness in leveraging local patterns within the data, making it ideal for handling feature spaces where classes might overlap. Its distance-based decision-making allows it to classify vehicle types by considering similarity to nearby samples/data. Additionally, it offers flexibility through hyperparameter tuning, such as the number of neighbors and distance metrics (as you see below).
- Hyperparameters Before Tuning:
  - Number of Neighbors: 5 (default)
  - Weight Function: Uniform (default)
  - Metric: Minkowski (default)
  - Algorithm: Auto (default)
    Note that these values were not explicitly specified when creating our baseline KNN. A KNN in scikit-learn without any specified
    hyperparameters (which is what we did for our baseline) will default to the values above.
- Hyperparameters After Tuning (Grid Search):
  - Number of Neighbors: 3
  - Weight Function: Distance
  - Metric: Euclidean
  - Algorithm: Auto
  ```python
   # testing different hyperparameters
   param_grid = {
       'n_neighbors': [3, 5, 7, 10, 12, 15],  
       'weights': ['uniform', 'distance'],  
       'metric': ['euclidean', 'manhattan'],
       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
   }
   
   # grid search cross-validation to find best hyperparameters for KNN
   grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5, scoring = 'accuracy', return_train_score = True)
   grid_search.fit(X_train_scaled, y_train)
   
   # get best hyperparameters
   best_params = grid_search.best_params_
  ```
Why These Parameters: After performing Grid Search Cross-Validation, these hyperparameters for KNN were selected for their optimal performance in classifying vehicle types. A lower number of neighbors (3) was chosen to allow the model to focus on more localized patterns, which is useful for distinguishing subtle differences between vehicle classes. The distance-based weight function prioritized closer neighbors, giving them more influence in the classification. The Euclidean distance metric was selected as it effectively measures straight-line similarity in the ResNet 50 embedding space, and the auto algorithm allowed scikit-learn to choose the most efficient method for nearest-neighbor searches based on the dataset size and structure.
<br>
<br>
Take a look at how other combinations of hyperparameters performed, and see any signs of underfitting/overfitting:
![image](https://github.com/user-attachments/assets/41a6a2e0-adc7-4b23-a057-66e4815f4a37)


## 4. **Results** :mag_right:

### 4.1 **Data Exploration Results**
- Class Distribution: 4 Classes, 100 Images Per Class, 400 Total. See Distribution Below (can also see in [Figures](#figures)):
![image](https://github.com/user-attachments/assets/12f06076-98b8-41e6-af1f-5ab455e56578)
- Number of Image Dimensions: 82 unique for Bus, 71 for Car, 74 for Motorcycle, and 78 for Truck. We did this to see if standardizing was necessary (it was!).
  ```
   Number of unique image dimensions for Bus: 82
   Number of unique image dimensions for Car: 71
   Number of unique image dimensions for motorcycle: 74
   Number of unique image dimensions for Truck: 78
  ```
- Example Images of Each Class to See Our Data (can also see in [Figures](#figures)):
![image](https://github.com/user-attachments/assets/c1efbd24-b440-4694-aa10-3f85d2766278)

### 4.2 **Preprocessing & Data Augmentation Results**
The results of our preprocessing and data augmentation steps can be visualized in the [Figures](#figures) section. Here are the highlights:
- Data Uniformity: All images were resized to 224x224 pixels, standardizing their dimensions
- Diversity Through Augmentation: Grayscale conversion, rotation, and horizontal flipping enhanced the dataset's variability.
- Feature Extraction by Converting Images into High-Dimensional Embeddings (using ResNet 50). See how we generated them in [Section 3.2](#embeddings), under "Feature Extraction".

### 4.3 **Model 1: Multi-Class SVM**
_Non-Tuned Model Results_ <br>
Training Set:
```
Error for Training Set: 0.0
Training Accuracy for SVM: 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       476
           1       1.00      1.00      1.00       481
           2       1.00      1.00      1.00       499
           3       1.00      1.00      1.00       464

    accuracy                           1.00      1920
   macro avg       1.00      1.00      1.00      1920
weighted avg       1.00      1.00      1.00      1920
```
Testing Set:
```
Error for Testing Set: 0.015
Testing Accuracy for SVM: 0.9854166666666667
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       124
           1       1.00      0.97      0.99       119
           2       0.97      0.98      0.98       101
           3       0.99      1.00      1.00       136

    accuracy                           0.99       480
   macro avg       0.98      0.98      0.98       480
weighted avg       0.99      0.99      0.99       480
```
_Tuned Model Results_ <br>
Training Set:
```
Error for Training Set (Tuned SVM): 0.0
Training Accuracy for Tuned SVM: 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       476
           1       1.00      1.00      1.00       481
           2       1.00      1.00      1.00       499
           3       1.00      1.00      1.00       464

    accuracy                           1.00      1920
   macro avg       1.00      1.00      1.00      1920
weighted avg       1.00      1.00      1.00      1920
```
Testing Set:
```
Error for Testing Set (Tuned SVM): 0.015
Testing Accuracy for Tuned SVM: 0.9854166666666667
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       124
           1       1.00      0.97      0.99       119
           2       0.97      0.98      0.98       101
           3       0.99      1.00      1.00       136

    accuracy                           0.99       480
   macro avg       0.98      0.98      0.98       480
weighted avg       0.99      0.99      0.99       480
```
Confusion Matrix (tuned SVM): <br>
![image](https://github.com/user-attachments/assets/095303fc-5005-4b97-97fc-de447278bfe9)

### 4.4 **Model 2: KNN**
_Non-Tuned Model Results_ <br>
Training Set:
```
Error for Training Set: 0.014
Training Accuracy for Baseline KNN: 0.9859375
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       476
           1       0.98      1.00      0.99       481
           2       0.99      0.96      0.98       499
           3       1.00      0.99      1.00       464

    accuracy                           0.99      1920
   macro avg       0.99      0.99      0.99      1920
weighted avg       0.99      0.99      0.99      1920
```
Testing Set:
```
Error for Testing Set: 0.037
Testing Accuracy for Baseline KNN: 0.9625
              precision    recall  f1-score   support

           0       0.94      0.97      0.95       124
           1       0.95      0.99      0.97       119
           2       0.96      0.88      0.92       101
           3       1.00      0.99      1.00       136

    accuracy                           0.96       480
   macro avg       0.96      0.96      0.96       480
weighted avg       0.96      0.96      0.96       480
```
_Tuned Model Results_ <br>
Training Set:
```
Error for Training Set: 0.0
Training Accuracy for Tuned KNN: 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       476
           1       1.00      1.00      1.00       481
           2       1.00      1.00      1.00       499
           3       1.00      1.00      1.00       464

    accuracy                           1.00      1920
   macro avg       1.00      1.00      1.00      1920
weighted avg       1.00      1.00      1.00      1920
```
Testing Set:
```
Error for Testing Set: 0.029
Testing Accuracy for Tuned KNN: 0.9708333333333333
              precision    recall  f1-score   support

           0       0.95      0.98      0.96       124
           1       0.96      1.00      0.98       119
           2       0.97      0.90      0.93       101
           3       1.00      0.99      1.00       136

    accuracy                           0.97       480
   macro avg       0.97      0.97      0.97       480
weighted avg       0.97      0.97      0.97       480


```
Confusion Matrix (tuned KNN): <br>
![image](https://github.com/user-attachments/assets/48865307-c58e-44e7-b1f3-293c0856cd9b)

## 5. **Discussion** :speaking_head:
Our project’s journey into vehicle type recognition began with a simple goal: to classify vehicle images accurately into their respective categories (Car, Motorcycle, Bus, Truck). From the outset, our approach involved thorough data exploration, preprocessing steps, and the implementation of various machine learning models to address this problem effectively.

### 5.1 **Data Exploration, Preprocessing, Augmentation**
The exploration phase highlighted the importance of standardizing the dataset. The raw dataset had inconsistencies in image dimensions, which could have hampered model performance. By resizing all images to 224x224 pixels, we ensured consistency across inputs. Data augmentation techniques, including grayscale conversion, rotation, and flipping, diversified our dataset and increased its robustness to variations in real-world scenarios like lighting and orientation. The embeddings generated using ResNet 50 provided a high-dimensional feature space that was crucial for the models we used.
<br>
<br>
However, one limitation was the relatively small dataset size initially before data augmentation (400 images across four classes). This would limit the models' ability to generalize well, especially for real-world applications where diverse and dynamic environments exist. Additionally, the augmented dataset may still not fully mimic real-world variations like partial occlusions, weather effects, extreme angles, and/or background noise. Regardless, these steps are crucial to ensure the models we built focus on relevant features, enhancing their ability to classify vehicle types accurately under controlled conditions and somewhat in real-world applicability.

### 5.2 **Model Implementation and Results**
#### **Multi-Class Support Vector Machine (SVM)**:
The Support Vector Machine (SVM) with a linear kernel proved to be a robust baseline. Its ability to handle high-dimensional ResNet embeddings efficiently led to near-perfect classification results. The tuned hyperparameters, including a lower regularization parameter (`C = 0.1`), allowed the SVM to generalize well without noticeable overfitting. The results were believable, as the high accuracy on both training and testing datasets aligned with the theoretical strengths of SVM in high-dimensional spaces. It's very likely that the exceptional performance of our SVM, tuned or not, comes from the inherent separability of the ResNet embeddings, as the pre-trained ResNet 50 model (a state-of-the-art model) had already captured key discriminative features for vehicle types. Additionally, the linear kernel effectively exploited this structured embedding space, where classes were well-distributed. This allowed for clear decision boundaries without requiring complex transformations during training.
#### **K-Nearest Neighbors (KNN)**:
The K-Nearest Neighbors (KNN) model, while effective in leveraging local patterns, slightly/marginally underperformed compared to SVM. The tuned parameters (e.g. `n_neighbors=3`, Euclidean distance) optimized its performance, but the sensitivity of KNN to overlapping feature spaces (especially in higher dimensions) led to minor misclassification / false negatives. This was evident in its slightly lower recall values for specific classes as seen in our results. The KNN model's performance was also affected by the high-dimensional nature of the ResNet embeddings, which can diminish the distance-based distinctions between classes ("curse of dimensionality"), therefore making the model more susceptible to noise and less robust. While its simplicity made it easy to implement and interpret, the computational cost and time of finding nearest neighbors in a large, high-dimensional dataset proved to be a significant drawback compared to our SVM approach. Despite this, KNN’s simplicity and interpretability made it a valuable comparison model.
### 5.3 **Shortcomings and Future Challenges**
One significant shortcoming was the limited dataset size and variability. Real-world images often contain noise and unpredictable variations, which were not fully captured in our dataset or augmentations. Additionally, our models relied on pre-extracted embeddings from ResNet 50 rather than training a custom deep learning model, which could potentially capture even finer details specific to our classification task. However, that does not take away from the fact that ResNet 50 is indeed a very strong, state-of-the-art CNN model best suited for image classification tasks like our project!
<br>
<br>
Another limitation was the computational efficiency of KNN during prediction. While it performed reasonably well, its reliance on storing and comparing all samples limits its scalability for larger datasets. Future solutions could involve integrating dimensionality reduction techniques such as PCA or SVD or switching to more scalable algorithms.
<br>
<br>
Lastly, while our evaluation metrics demonstrated strong performance, they do not account for edge cases or misclassifications in real-world scenarios. For example, how would the model perform if the dataset included ambiguous images (e.g., buses with advertisements that make them look like trucks, trucks that look like cars, or classifying a specific car in traffic filled with other cars and various noise)? These scenarios remain unexplored and pose challenges for practical deployment.

## 6. **Conclusion** :checkered_flag:
Our final project highlights the potential of machine learning models for solving practical classification problems like vehicle type recognition. The SVM model emerged as the most effective approach due to its computational efficiency and strong performance in high-dimensional feature spaces. However, the KNN model also provided valuable insights into local patterns and alternative classification strategies, albeit inefficient and slow as we were with high-dimensional data (ResNet 50 embeddings).
<br>
<br>
In hindsight, several improvements could have enhanced our results:
- **Dataset Expansion**: Collecting a larger, more diverse dataset with real-world variations could improve the model’s robustness.
- **Deep Learning Models**: Training a convolutional neural network (CNN) from scratch or fine-tuning a pre-trained model could capture more task-specific features.
- **Context-Aware Classification**: Incorporating contextual information, such as vehicle surroundings or traffic conditions, could further refine our classification task.
- **Exploration of Other Models**: Implementing ensemble methods or more advanced algorithms like Random Forests or Gradient-Boosted Trees could provide alternative classification strategies.

In conclusion, our project demonstrates the strengths and limitations of different models while emphasizing the importance of thoughtful preprocessing and model selection. While our results were promising, they are only the beginning. Future work should focus on addressing the challenges posed by real-world variability, scaling solutions, and exploring innovative, modern, state-of-the-art methods to push the boundaries and further enhance vehicle type recognition.

## 6. **Statement of Collaboration** :clap:
**Team Member 1**: <br>
Name: Adrian Apsay <br>
Title: Project Team Lead <br>
Contribution: Laid out plans for each milestone up until the final submission. Managed and communicated with team members accordingly. Major code contributor. Major write-up contributor.
<br>
<br>
**Team Member 2**: <br>
Name: <br>
Title: <br>
Contribution: 
<br>
<br>
**Team Member 3**: <br>
Name: <br>
Title: <br>
Contribution: 
<br>
<br>
**Team Member 4**: <br>
Name: <br>
Title: <br>
Contribution: 
<br>
<br>
**Team Member 5**: <br>
Name: <br>
Title: <br>
Contribution: 
<br>
<br>
**Team Member 6**: <br>
Name: <br>
Title: <br>
Contribution: 
