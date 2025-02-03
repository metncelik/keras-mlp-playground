from utils import save_code, get_lines
from keras import layers as l
from keras import models as m
import numpy as np
from keras.utils import to_categorical
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random
import glob

# set random seeds for reproducibility
RANDOM_SEED = 196
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# hyperparameters
LEARNING_RATE = 0.0005
LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=True)
EPOCHS = 150
BATCH_SIZE = 32

VALIDATION_RATE = 0.2

# create timestamp for folder name
timestamp = datetime.now().strftime('%d-%H-%M-%S')
output_dir = f"runs/{timestamp}"

# create directory structure
os.makedirs(f"{output_dir}/model", exist_ok=True)
os.makedirs(f"{output_dir}/plot", exist_ok=True)
os.makedirs(f"{output_dir}/code", exist_ok=True)
os.makedirs(f"{output_dir}/summary", exist_ok=True)

# Get list of available datasets
dataset_files = glob.glob("dataset/*.csv")
selected_dataset = None

if len(dataset_files) > 1:
    print("\nAvailable datasets:")
    for i, dataset_file in enumerate(dataset_files):
        print(f"{i+1}. {os.path.basename(dataset_file)}")
    
    while True:
        try:
            choice = int(input("\nSelect a dataset (enter number): ")) - 1
            if 0 <= choice < len(dataset_files):
                selected_dataset = dataset_files[choice]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
else:
    selected_dataset = dataset_files[0]

print(f"\nUsing dataset: {os.path.basename(selected_dataset)}")
dataset = np.loadtxt(selected_dataset, delimiter=",")

# split data
size = len(dataset)
split_rate = 0.8
pivot = int(size * split_rate)

# split data
train_datas = dataset[:pivot, :10]
test_datas = dataset[pivot:, :10]

train_labels = dataset[:pivot, 10]
test_labels = dataset[pivot:, 10]

# check class distribution and compute class weights
class_counts = np.bincount(train_labels.astype(int))
class_weights = {i: (len(train_labels) / (3 * count)) for i, count in enumerate(class_counts)}

# normalize the input features
scaler = StandardScaler()
train_datas = scaler.fit_transform(train_datas)
test_datas = scaler.transform(test_datas)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# create model with improved architecture
model = m.Sequential()
model.add(l.Dense(128, input_dim=10, activation='celu'))
model.add(l.Dense(96, activation='elu'))
model.add(l.Dense(64, activation='selu'))
model.add(l.Dense(48, activation='gelu'))
model.add(l.Dense(32, activation='tanh'))
model.add(l.Dense(16, activation='swish'))
model.add(l.Dense(8, activation='relu'))
model.add(l.Dense(3, activation='softplus'))

model.compile(loss=LOSS_FUNCTION, 
              optimizer=OPTIMIZER, 
              metrics=["accuracy"])

# callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
model_checkpoint = ModelCheckpoint(
    f'{output_dir}/model/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
callbacks = [early_stopping, model_checkpoint]

history = model.fit(
    train_datas, 
    train_labels, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_RATE,
    callbacks=callbacks,
    class_weight=class_weights
)

# plot training history
plt.figure(figsize=(12, 4))

# plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

#plt.show()

# save plot
plt.savefig(f"{output_dir}/plot/training_history.png")
plt.close()

train_eval = model.evaluate(train_datas, train_labels)
test_eval = model.evaluate(test_datas, test_labels)


#change dir name to timestamp + metrics
new_output_dir = f"runs/{timestamp}-train-{int(train_eval[1]*100)}-test-{int(test_eval[1]*100)}"
os.rename(output_dir, new_output_dir)

# save code with metrics in filename 
save_code("main.py", f"{new_output_dir}/code/model-{int(train_eval[1]*100)}-{int(test_eval[1]*100)}.py")

print(model.summary())
print("\nTraining Configuration:")
print("-----------------------")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Loss Function: {LOSS_FUNCTION}")
print(f"Optimizer: {OPTIMIZER.__class__.__name__}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Split rate: {split_rate}")
print(f"Split pivot: {pivot}")
print(f"Validation rate: {VALIDATION_RATE}")
print(f"(train) {model.metrics_names[1]}: %{train_eval[1]*100:.2f}")
print(f"(test) {model.metrics_names[1]}: %{test_eval[1]*100:.2f}")

# create a summary file
with open(f"{new_output_dir}/summary/summary.txt", "w") as f:
    f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Dataset:\n")
    f.write("---------\n")
    f.write(f"Selected dataset: {os.path.basename(selected_dataset)}\n\n")
    
    f.write("Model Configuration:\n")
    f.write("-------------------\n")
    f.write(f"Random Seed: {RANDOM_SEED}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Loss Function: {LOSS_FUNCTION}\n")
    f.write(f"Optimizer: {OPTIMIZER.__class__.__name__}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n\n")
    
    f.write("Training Parameters:\n")
    f.write("-------------------\n")
    f.write(f"Split rate: {split_rate}\n")
    f.write(f"Split pivot: {pivot}\n")
    f.write(f"Validation rate: {VALIDATION_RATE}\n\n")
    
    f.write("Results:\n")
    f.write("--------\n")
    f.write(f"Training accuracy: {train_eval[1]*100:.2f}%\n")
    f.write(f"Test accuracy: {test_eval[1]*100:.2f}%\n\n")
    
    f.write("Model:\n")
    f.write("------\n")
    f.write(f"{get_lines('main.py', 'model =', '# plot training history')}\n")