#Step - 1 Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#Step - 2 Constants
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
NUM_CLASSES = 3
IMAGE_SIZE = (256,256)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "models/best_model.keras"

#Step - 3 Load Dataset **seperate tf dataset object for dl from train and val directories**
train_dataset = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE
)

validation_dataset = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE
)

#Step - 4 **Optimize Dataset Pipeline- tf loads next batch while current batch is being processed**
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#Step-5 Data Augmentation
data_augmentation =keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

#Step -6 ***CNN Model builing***
model = keras.Sequential([
    keras.Input(shape=(256,256,3)),
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    
    layers.Dense(128, activation ="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation = "softmax")   
])

#Step - 7 Compile Model
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Step -8 Configure Callback
checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor = "val_accuracy",
    save_best_only = True

)

#Step -9 Train Model 
history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = EPOCHS,
    callbacks=[checkpoint]
)
#Step 10 Best Model already saved by ModelCheckpoint 

#Step - 11 Evaluate Model
loss, accuracy = model.evaluate(validation_dataset)
print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

#Step -12 Plot Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

#Step -13 Plot Loss Graph
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.show()

