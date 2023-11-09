
#NVIDIA AI Course Notes

from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)

#output
#Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
#58892288/58889256 [==============================] - 1s 0us/step

# Freeze base model
#This is done so that all the learning from the ImageNet dataset does not get destroyed in the initial training.
base_model.trainable = False


# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(1, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

model.summary()

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
vgg16 (Model)                (None, 7, 7, 512)         14714688  
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 513       
=================================================================
Total params: 14,715,201
Trainable params: 513
Non-trainable params: 14,714,688
_________________________________________________________________

#Now it's time to compile the model with loss and metrics options. Remember that we're training on a number of different categories, rather than a binary classification problem.
model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.CategoricalAccuracy()])


#If you'd like, try to augment the data to improve the dataset.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False, # Don't randomly flip images vertically
)  
datagen_valid = ImageDataGenerator(
    rescale=1.0/255
)

## Load Dataset

#Now it's time to load the train and validation datasets. Pick the right folders, as well as the right target_size of the images (it needs to match the height and width input of the model you've created). 

# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    "data/fruits",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="binary",
    batch_size=8,
)

# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    "data/fruits",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="binary",
    batch_size=8,
)

#output: Found 1511 images belonging to 2 classes.
#Found 1511 images belonging to 2 classes.

model.fit(train_it, steps_per_epoch=12, validation_data=valid_it, validation_steps=4, epochs=20)

Epoch 1/20
12/12 [==============================] - 1s 118ms/step - loss: 0.0000e+00 - categorical_accuracy: 1.0000 - val_loss: 0.0000e+00 - val_categorical_accuracy: 1.0000

#in between 2-19, all similar in this instance. not usual case. usually there are error differences i.e., .94, .1, .85

Epoch 20/20
12/12 [==============================] - 1s 119ms/step - loss: 0.0000e+00 - categorical_accuracy: 1.0000 - val_loss: 0.0000e+00 - val_categorical_accuracy: 1.0000


##########################################OPTIONAL: ##########################################################
#Fine tune learning rate with this to a LOW learning rate in order to not overfit the AI model

# Unfreeze the base model
#base_model.trainable = FIXME
'''
# Compile the model with a low learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = FIXME),
              loss = FIXME , metrics = FIXME)
model.fit(FIXME,
          validation_data=FIXME,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=FIXME)

#Evaluate model will return a tuple, first value is loss, seconmd is accuracy
model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)
'''
##########################################End of Optional: ##########################################################

from run_assessment import run_assessment

run_assessment(model, valid_it)

Evaluating model 5 times to obtain average accuracy...

189/188 [==============================] - 10s 53ms/step - loss: 0.0000e+00 - categorical_accuracy: 1.0000
189/188 [==============================] - 10s 53ms/step - loss: 0.0000e+00 - categorical_accuracy: 1.0000
189/188 [==============================] - 10s 54ms/step - loss: 0.0000e+00 - categorical_accuracy: 1.0000
189/188 [==============================] - 10s 55ms/step - loss: 0.0000e+00 - categorical_accuracy: 1.0000
189/188 [==============================] - 10s 55ms/step - loss: 0.0000e+00 - categorical_accuracy: 1.0000

Accuracy required to pass the assessment is 0.92 or greater.
Your average accuracy is 1.0000.

Congratulations! You passed the assessment!
See instructions below to generate a certificate.

