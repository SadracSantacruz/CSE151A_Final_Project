# Vehicle Type Recognition Project

## 1. **Introduction** :car:
The goal of our project was to develop a machine learning system capable of recognizing vehicle types (Car, Motorcycle, Bus, and Truck) based on images. This problem was chosen due to its practical applications in areas such as traffic management, road infrastructure, and smarter, efficient cities.

A robust, high-performing predictive model in this domain could revolutionize how vehicles are monitored and classified, leading to smarter infrastructure planning and improved road safety. Such models could be integrated into intelligent traffic cameras for the following use cases:
- **Traffic Trends**: Classifying vehicles in real-time allows authorities to gather data on the types of vehicles using specific roads at different times (buses during rush hour, trucks at night, etc.). It also can identify what cars exist in different hours of traffic, high-usage vehicle types at a certain hour, etc. which can be used to develop better traffic management strategies.
- **Road Layouts**: Vehicle classification could be used to retrieve what roads are frequently used by certain vehicles. An example of this would be heavy trucks on certain roads, calling for stronger pavement or wider lanes. This type of classification can also be used to optimize traffic lights (i.e. what kind of vehicles are passing by a certain light?)
- **Public Transport System Design**: Classification can help with what routes are car/truck/motorcycle-dominated, and what routes have a low utilization of buses. This can help guide where public transit systems should be expanded, adjusted, or even implemented. It can also help estimate the reliance on personal vehicles versus public transport, being able to identify areas where transit investments can reduce congestion and emissions.
<br>
This project was also a fun execution of our interests as a group, as we all particularly love cars.

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

## 3. **Methods**

### 3.1 **Data Exploration**
- Dataset: Vehicle Type Recognition Dataset from Kaggle
- Exploratory Steps:
  - Visualized class distribution (evenly distributed, see "Figures")
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
  - Visualized example images for each class (see "Figures")

### 3.2 **Preprocessing & Data Augmentation** (see these preprocessed image examples in "**Figures**" Section)
- **Image Scaling**: Resized all images to 224x224 pixels.
- **Grayscaling**: Converted images to grayscale.
- **Rotation**: Applied rotations of 15°, 30°, 45°, 60°, and 75°.
- **Flipping**: Horizontally flipped images.
- **Feature Extraction**: Used ResNet50 to generate embeddings (to feed into models that require numerical representation!). See how we generated these embeddings (code below):
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
Justification: We chose this model due to its effectiveness in handling high-dimensional data (in this case, our ResNet50 embeddings) and its ability to create clear decision boundaries for classification tasks. By leveraging a linear kernel and the One-vs-Rest approach, it provided a robust baseline for separating vehicle types based on the extracted feature embeddings. Its computational efficiency during training and prediction made it a suitable choice for this problem.
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
Why These Parameters: We decided to use these tuned hyperparameters for our SVM after Grid Search as it fetched the performance out of all combinations. A linear kernel was chosen because the ResNet50 embeddings are high-dimensional, and a linear boundary is often effective for such data. The One-vs-Rest approach allows the SVM to handle multi-class classification by training a separate binary classifier for each class. A lower C value of 0.1 was selected to prevent overfitting by allowing a softer margin and better generalization.

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
Why These Parameters: After performing Grid Search, these hyperparameters for KNN were selected for their optimal performance in classifying vehicle types. A lower number of neighbors (3) was chosen to allow the model to focus on more localized patterns, which is useful for distinguishing subtle differences between vehicle classes. The distance-based weight function prioritized closer neighbors, giving them more influence in the classification. The Euclidean distance metric was selected as it effectively measures straight-line similarity in the ResNet50 embedding space, and the auto algorithm allowed scikit-learn to choose the most efficient method for nearest-neighbor searches based on the dataset size and structure.

## 4. **Results**

### 4.1 **Data Exploration Results**
- Number of Classes: 4
- Total Number of Images: 400
- Example Image Dimensions: 82 unique for Bus, 71 for Car, 74 for Motorcycle, and 78 for Truck.

### 4.2 **Baseline Model: Multi-Class SVM**
- Training Accuracy: 1.0
- Testing Accuracy: 98.5%
- Confusion Matrix:
  - Class 0: Precision = 0.98, Recall = 0.98
  - Class 1: Precision = 1.00, Recall = 0.97

### 4.3 **Final Model: KNN**
- Training Accuracy: 98.6%
- Testing Accuracy: 96.2%
- Confusion Matrix:
  - Class 0: Precision = 0.94, Recall = 0.97
  - Class 1: Precision = 0.95, Recall = 0.99

## 5. **Discussion**

### 5.1 **Data Exploration and Preprocessing**
The preprocessing steps standardized the dataset, enabling consistent input for the models. Data augmentation (scaling, flipping, etc.) effectively increased the dataset size and diversity.

### 5.2 **Baseline Model**
The SVM baseline demonstrated remarkable accuracy with minimal error. Its ability to perfectly classify training data highlighted its suitability for high-dimensional feature spaces.

### 5.3 **Final Model**
The KNN model, while slightly less accurate than SVM, provided valuable insights into local classification strategies. However, it struggled with classes having overlapping feature spaces, as evidenced by slightly lower recall values.

### 5.4 **Model Comparisons**
The SVM’s linear kernel offered robust generalization, outperforming KNN in terms of both accuracy and computational efficiency during prediction.

### 5.5 **Shortcomings**
- Limited dataset size capped the models’ potential.
- Real-world images might introduce complexities such as occlusions or varying lighting conditions not captured in the dataset.

## 6. **Conclusion**
The project showcased the effectiveness of machine learning models in vehicle type recognition. The SVM’s superior accuracy and efficiency made it the final choice. Future work could involve:
- Using larger, more diverse datasets.
- Exploring deep learning models like Convolutional Neural Networks (CNNs).
- Fine-tuning augmentation techniques to better mimic real-world conditions.

In conclusion, this project highlighted the potential of machine learning in solving real-world classification problems while identifying avenues for improvement and future exploration.

