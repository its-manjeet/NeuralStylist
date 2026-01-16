import tensorflow as tf
from tensorflow import keras

# 1. DOWNLOAD DATA (Fashion MNIST - 70,000 images)
print("📥 Downloading Fashion Data...")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values (0-255 -> 0-1)
train_images = train_images / 255.0

# 2. BUILD THE MODEL (The Pro Brain)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),       # Input: 28x28 image
    keras.layers.Dense(128, activation='relu'),       # Thinking Layer
    keras.layers.Dense(10, activation='softmax')      # Output: 10 clothing types
])

# 3. COMPILE
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRAIN (Takes about 30 seconds)
print("🧠 Training Neural Stylist...")
model.fit(train_images, train_labels, epochs=5)

# 5. SAVE
model.save('fashion_brain.h5')
print("✅ DONE! Saved 'fashion_brain.h5'. You are ready for Step 2.")