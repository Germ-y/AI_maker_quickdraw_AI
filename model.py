import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Input

name="teddy-bear"

classes = [name,'necklace']
X_list = []
y_list = []

for idx, cls in enumerate(classes):
    data = np.load(f"datasets/full_numpy_bitmap_{cls}.npy")
    X_list.append(data)
    y_list.append(np.full(data.shape[0], idx))

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes=len(classes))

num_classes = len(classes)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

model.save(f"quickdraw_class_model_{name}.keras")

with open(f"categories_{name}.txt", "w") as f:
    for cls in classes:
        f.write(cls + "\n")