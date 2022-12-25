import tensorflow_addons as tfa
import tensorflow as tf
import argparse
import os
from utils.load_datasets import GenerateDatasets
from model.model_builder import ModelBuilder
from model.loss import DepthEstimationLoss


class ModelConfiguration(GenerateDatasets):
    def __init__(self, args: argparse, mirrored_strategy: object = None):
        """
        Args:
            args (argparse): Training options (argparse).
            mirrored_strategy (tf.distribute): tf.distribute.MirroredStrategy() with Session.
        """
        self.args = args
        self.mirrored_strategy = mirrored_strategy
        self.check_directory(dataset_dir=args.dataset_dir,
                             checkpoint_dir=args.checkpoint_dir, model_name=args.model_name)
        self.configuration_args()

        super().__init__(data_dir=self.DATASET_DIR,
                         image_size=self.IMAGE_SIZE,
                         batch_size=self.BATCH_SIZE,
                         dataset_name=args.dataset_name,
                         is_tunning=False,
                         percentage=100
                         )


    def check_directory(self, dataset_dir: str, checkpoint_dir: str, model_name: str) -> None:
        """
        Args:
            dataset_dir    (str) : Tensorflow dataset directory.
            checkpoint_dir (str) : Directory to store training weights.
            model_name     (str) : Model name to save.
        """
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir + model_name, exist_ok=True)
    
    def get_model_hyparameter_prefix(self, args) -> str:
        batch_size = str(args.batch_size)
        epoch = str(args.epoch)
        lr = str(args.lr)
        image_size = str(args.image_size[0])
        optimizer = str(args.optimizer)
        
        if args.multi_gpu:
            gpu_type = 'multi-gpu'
        else:
            gpu_type = 'single-gpu'
            
        prefix = 'Bs-{0}_Ep-{1}_Lr-{2}_ImSize-{3}_Opt-{4}_{5}'.format(batch_size,
                                                                   epoch,
                                                                   lr,
                                                                   image_size,
                                                                   optimizer,
                                                                   gpu_type)
        return prefix

    def configuration_args(self):
        """
            Set training variables from argparse's arguments 
        """
        self.MODEL_PREFIX = self.args.model_prefix
        self.WEIGHT_DECAY = self.args.weight_decay
        self.OPTIMIZER_TYPE = self.args.optimizer
        self.BATCH_SIZE = self.args.batch_size
        self.EPOCHS = self.args.epoch
        self.INIT_LR = self.args.lr
        self.SAVE_MODEL_NAME = self.get_model_hyparameter_prefix(self.args) + '_' + self.args.model_name + '_' +  self.MODEL_PREFIX
        self.DATASET_DIR = self.args.dataset_dir
        self.DATASET_NAME = self.args.dataset_name
        self.CHECKPOINT_DIR = self.args.checkpoint_dir
        self.TENSORBOARD_DIR = self.args.tensorboard_dir
        self.IMAGE_SIZE = self.args.image_size
        self.USE_WEIGHT_DECAY = self.args.use_weight_decay
        self.MIXED_PRECISION = self.args.mixed_precision
        self.DISTRIBUTION_MODE = self.args.multi_gpu
        if self.DISTRIBUTION_MODE:
            self.BATCH_SIZE *= 2

    def configuration_dataset(self) -> None:
        """
            Configure the dataset. Train and validation dataset is inherited from the parent class and used.
        """
        # Wrapping tf.data generator
        self.train_data = self.get_trainData(train_data=self.train_data)
        self.valid_data = self.get_validData(valid_data=self.valid_data)
    
        # Calculate training and validation steps
        self.steps_per_epoch = self.number_train // self.BATCH_SIZE
        self.validation_steps = self.number_valid // self.BATCH_SIZE

        # Wrapping tf.data generator if when use multi-gpu training
        if self.DISTRIBUTION_MODE:
            self.train_data = self.mirrored_strategy.experimental_distribute_dataset(self.train_data)
            self.valid_data = self.mirrored_strategy.experimental_distribute_dataset(self.valid_data)   

    def __set_callbacks(self):
        """
            Set the keras callback of model.fit.

            For some metric callbacks, the name of the custom metric may be different and may not be valid,
            so you must specify the name of the custom metric.
        """
        # Set training keras callbacks
        checkpoint_val_loss = tf.keras.callbacks.ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name + '/_' + self.SAVE_MODEL_NAME + '_best_loss.h5',
                                              monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

        checkpoint_metric = tf.keras.callbacks.ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name + '/_' + self.SAVE_MODEL_NAME + '_best_rmse.h5',
                                              monitor='val_root_mean_squared_error', save_best_only=True, save_weights_only=True, verbose=1)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.TENSORBOARD_DIR + 'train/' +
                                  self.MODEL_PREFIX, write_graph=True, write_images=True)

    
        polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.INIT_LR,
                                                                  decay_steps=self.EPOCHS,
                                                                  end_learning_rate=self.INIT_LR * 0.1, power=0.9)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay, verbose=1)
        
        # If you wanna need another callbacks, please add here.
        self.callback = [checkpoint_val_loss, checkpoint_metric,  tensorboard, lr_scheduler]
    
    def __set_optimizer(self):
        """
            Configure the optimizer for backpropagation calculations.
        """
        if self.OPTIMIZER_TYPE == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.INIT_LR, amsgrad=True)
        elif self.OPTIMIZER_TYPE == 'radam':
            self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.INIT_LR,
                                                          weight_decay=0.00001,
                                                          total_steps=int(
                                                          self.number_train / (self.BATCH_SIZE / self.EPOCHS)),
                                                          warmup_proportion=0.1,
                                                          min_lr=0.0001)
        elif self.OPTIMIZER_TYPE == 'adamW':
            self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=self.INIT_LR, weight_decay=0.0001, amsgrad=True)
        if self.MIXED_PRECISION:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            # Wrapping optimizer by mixed precision
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

    def __set_metrics(self):
        rmse_metric = tf.keras.metrics.RootMeanSquaredError()
        metrics = [rmse_metric]
        return metrics

    def __configuration_model(self):
        """
            Build a deep learning model.
        """
        # Get instance model builder
        model_builder = ModelBuilder(image_size=self.IMAGE_SIZE,
                                  use_weight_decay=self.USE_WEIGHT_DECAY,
                                  weight_decay=self.WEIGHT_DECAY,
                                  is_tunning=False)

        # Build model by model name
        model = model_builder.build_model()
        model.summary()

        return model

    def train(self):
        """
            Compile all configuration settings required for training.
            If the custom metric name is different in the __set_callbacks function,
            the update may not be possible, so please check the name.
        """
        self.configuration_dataset()
        self.metrics = self.__set_metrics()
        self.__set_optimizer()
        self.__set_callbacks()
        self.model = self.__configuration_model()
  
        
        # loss_obj= Total_loss(num_classes=self.num_classes)
        # self.loss = loss_obj.detection_loss

        self.loss = DepthEstimationLoss(global_batch_size=self.BATCH_SIZE, distribute_mode=self.DISTRIBUTION_MODE).depth_loss

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        # self.model.summary()

        self.model.fit(self.train_data,
                       validation_data=self.valid_data,
                       steps_per_epoch=self.steps_per_epoch,
                       validation_steps=self.validation_steps,
                       epochs=self.EPOCHS,
                       callbacks=self.callback)


    def saved_model(self):
        """
            Convert it to a graph model (.pb) using the learned weights.
        """
        self.model = ModelBuilder(image_size=self.IMAGE_SIZE,
                                  num_classes=self.num_classes).build_model()
        self.model.load_weights(self.args.saved_model_path)
        export_path = os.path.join(self.CHECKPOINT_DIR, 'export_path', '1')
        
        os.makedirs(export_path, exist_ok=True)
        self.export_path = export_path

        self.model.summary()

        tf.keras.models.save_model(
            self.model,
            self.export_path,
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None
        )
        print("save model clear")