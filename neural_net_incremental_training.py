import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Neural_Net:

    def __init__(self, dataset_folder_path, old_model_name, new_model_name):
        # Define your directories
        self.classes = ['DB', 'DT', 'HH', 'HL', 'LH', 'LL']
        self.directories = [dataset_folder_path + '/' + dir for dir in self.classes]
        self.check_directories(dataset_folder_path)

        # Define test size and batch size for preprocessing
        self.test_size = 0.2
        self.preprocess_batch_size = 100

        # Define output directory for preprocessed batches
        self.output_dir = 'preprocessed_data'
        # Define output directory for preprocessed batches
        self.preprocessed_train_dir = os.path.join(self.output_dir, 'train')
        self.preprocessed_test_dir = os.path.join(self.output_dir, 'test')

        # Define input shape and number of classes
        self.input_shape = (669, 40, 3)
        self.num_classes = 6

        # Define batch size
        self.batch_size = 32

        # Define model names
        self.old_model_name = old_model_name
        self.new_model_name = new_model_name

    def check_directories(self, dataset_folder_path):
        """Check if the dataset folder contains the required classes and images."""
        # Check if the dataset folder exists
        if not os.path.exists(dataset_folder_path):
            print('Dataset folder not found.')
            sys.exit()
        # Check if the dataset folder contains the required classes
        for dir in self.directories:
            if not os.path.exists(dir):
                print(f'Class folder {dir} not found.')
                sys.exit()
        # Check if the dataset folder contains images
        for dir in self.directories:
            if len(os.listdir(dir)) == 0:
                print(f'No images found in class folder {dir}.')
                sys.exit()

    def preprocess_data(self):
        for i, dir in enumerate(self.directories):
            images = []
            labels = []

            # Loop through each file in the directory
            for filename in os.listdir(dir):
                # Ensure the file is an image
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # Open the image file
                    img = Image.open(os.path.join(dir, filename))
                    # Convert the image to a numpy array and append it to the images list
                    images.append(np.array(img))
                    # Append the corresponding label to the labels list
                    labels.append(i)

            # Convert lists to numpy arrays
            X = np.array(images)
            y = np.array(labels)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

            # Define batch size for preprocessing
            num_samples_train = len(X_train)
            num_batches_train = num_samples_train // self.preprocess_batch_size + 1 if num_samples_train % self.preprocess_batch_size != 0 else num_samples_train // self.preprocess_batch_size

            num_samples_test = len(X_test)
            num_batches_test = num_samples_test // self.preprocess_batch_size + 1 if num_samples_test % self.preprocess_batch_size != 0 else num_samples_test // self.preprocess_batch_size

            # Create output directories if they don't exist
            train_dir = os.path.join(self.output_dir, 'train', f'class_{i}')
            test_dir = os.path.join(self.output_dir, 'test', f'class_{i}')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Preprocess training data
            for i in range(num_batches_train):
                start_id_x = i * self.preprocess_batch_size
                end_id_x = min((i + 1) * self.preprocess_batch_size, num_samples_train)
                batch_X = X_train[start_id_x:end_id_x]

                for j in range(len(batch_X)):
                    np.save(os.path.join(train_dir, f'batch_{i * self.preprocess_batch_size + j}.npy'), batch_X[j])

            # Preprocess test data
            for i in range(num_batches_test):
                start_id_x = i * self.preprocess_batch_size
                end_id_x = min((i + 1) * self.preprocess_batch_size, num_samples_test)
                batch_X = X_test[start_id_x:end_id_x]

                for j in range(len(batch_X)):
                    np.save(os.path.join(test_dir, f'batch_{i * self.preprocess_batch_size + j}.npy'), batch_X[j])

    def train(self, old_model_name, new_model_name):
        model = tf.keras.models.load_model(old_model_name + '.h5')

        train_data = []
        train_labels = []

        for i in range(self.num_classes):
            class_train_data = []
            class_train_labels = []

            class_train_dir = os.path.join(self.preprocessed_train_dir, f'class_{i}')
            for file in os.listdir(class_train_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(class_train_dir, file))
                    class_train_data.append(data)
                    class_train_labels.append(i)

            train_data.extend(class_train_data)
            train_labels.extend(class_train_labels)

        train_X = np.array(train_data)
        train_y = np.array(train_labels)

        # Shuffle training data
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]

        # Reshape images to the correct shape
        train_X = train_X.reshape((-1,) + self.input_shape)

        # Create training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(len(train_X)).batch(self.batch_size)

        # Train the model
        model.fit(train_dataset, epochs=10)

        model.save(new_model_name + '.h5')


    def evaluate_model(self, model_name):
        # Load preprocessed data for testing
        test_data = []
        test_labels = []

        for i in range(self.num_classes):
            class_test_data = []
            class_test_labels = []

            class_test_dir = os.path.join(self.preprocessed_test_dir, f'class_{i}')
            for file in os.listdir(class_test_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(class_test_dir, file))
                    class_test_data.append(data)
                    class_test_labels.append(i)

            test_data.extend(class_test_data)
            test_labels.extend(class_test_labels)

        test_X = np.array(test_data)
        test_y = np.array(test_labels)

        # Shuffle testing data
        indices = np.arange(len(test_X))
        np.random.shuffle(indices)
        test_X = test_X[indices]
        test_y = test_y[indices]

        # Reshape images to the correct shape
        test_X = test_X.reshape((-1,) + self.input_shape)

        # Load preprocessed data for testing
        test_data = []
        test_labels = []

        for i in range(self.num_classes):
            class_test_data = []
            class_test_labels = []

            class_test_dir = os.path.join(self.preprocessed_test_dir, f'class_{i}')
            for file in os.listdir(class_test_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(class_test_dir, file))
                    class_test_data.append(data)
                    class_test_labels.append(i)

            test_data.extend(class_test_data)
            test_labels.extend(class_test_labels)

        test_X = np.array(test_data)
        test_y = np.array(test_labels)

        # Shuffle testing data
        indices = np.arange(len(test_X))
        np.random.shuffle(indices)
        test_X = test_X[indices]
        test_y = test_y[indices]

        # Reshape images to the correct shape
        test_X = test_X.reshape((-1,) + self.input_shape)
        # Load the model
        model = tf.keras.models.load_model(model_name + '.h5')

        # Make predictions
        predictions = model.predict(test_X)

        # Convert the one-hot encoded predictions to labels
        predictions = tf.argmax(predictions, axis=1)

        # Display the true labels and the predicted labels
        true_labels = test_y[:5]
        print('True labels:', true_labels)
        print('Predicted labels:', predictions[:5])
        # Calculate the accuracy of the model
        accuracy = np.mean(test_y == predictions)
        print('Accuracy:', accuracy)

        # Display the confusion matrix
        cm = confusion_matrix(test_y, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.classes, yticklabels=self.classes, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    def main(self):
        self.preprocess_data()
        self.train(self.old_model_name, self.new_model_name)
        self.evaluate_model(self.new_model_name)

if __name__ == '__main__':
    dataset_folder_path = sys.argv[1]
    old_model_name = sys.argv[2]
    new_model_name = sys.argv[3]
    nn = Neural_Net(dataset_folder_path, old_model_name, new_model_name)
    nn.main()
