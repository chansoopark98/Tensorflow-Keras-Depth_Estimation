import tensorflow as tf
from tensorflow import keras
from utils.load_datasets import GenerateDatasets
tf.keras.backend.clear_session()

image_size = (256, 256)
batch_size = 16
learning_rate = 0.001
epoch = 5
max_trials = 20
optimizer_type = 'adam'
weight_decay = 0.0001

model_name = 'csnet_tunning_test'

kernel_initializer = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


data_loader = GenerateDatasets(data_dir='./datasets/',
                                 image_size=image_size,
                                 batch_size=batch_size,
                                 dataset_name='nyu_depth_v2',
                                 is_tunning=True)

# Concatenate dataset (train+valid)
train_data = data_loader.train_data
valid_data = data_loader.valid_data

# Wrapping tf.data generator
train_data = data_loader.get_trainData(train_data=train_data)
valid_data = data_loader.get_validData(valid_data=valid_data)

steps_per_epoch = data_loader.number_train // batch_size
validation_steps = data_loader.number_valid // batch_size

# callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/' + 'detection_tunning/' +
                                  model_name, write_graph=True, write_images=True)

polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
                            decay_steps=epoch,
                            end_learning_rate=learning_rate * 0.9, power=0.5)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping('val_loss', patience=1)

callback = [tensorboard, lr_scheduler, early_stopping]

# optimizer
if optimizer_type == 'sgd':
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=learning_rate)
elif optimizer_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif optimizer_type == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=0.9)
else:
    raise print('unknown optimizer type')

accuracy_metric = tf.keras.metrics.MeanSquaredError()
metrics = [accuracy_metric]

# mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

input_tensor = tf.keras.Input(shape=(*image_size, 3))

from model.model_zoo.CSNet_HR_lite import CSNetHRLite
import keras_tuner

def build_model(hp: keras_tuner.HyperParameters()):
    from model.loss import DepthEstimationLoss
    model = CSNetHRLite(image_size=image_size, classifier_activation=None,                        
                        use_multi_gpu=False).build_model(hp)
    loss = DepthEstimationLoss(global_batch_size=batch_size).depth_loss

    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    return model


tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
                                         objective='val_loss',
                                         max_trials=max_trials,
                                         overwrite=True,
                                         directory='keras_tuner',
                                         project_name='csnet-depth-estimation'
                                         )
print('search model')
tuner.search(train_data, epochs=epoch, validation_data=valid_data, callbacks=callback)
tuner.search_space_summary()
tuner.results_summary()

best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)