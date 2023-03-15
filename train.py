"""
    Run command
    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py
"""
from model_configuration import ModelConfiguration
import tensorflow as tf
import argparse
import time

tf.keras.backend.clear_session()
tf.get_logger().setLevel('INFO')
parser = argparse.ArgumentParser()

# Set Training Options
parser.add_argument("--model_prefix",     type=str,    help="Model name (logging weights name and tensorboard)",
                    default='230315_EfficientV2B0_customLoss_480x640_adam_lossFactor_test2')
parser.add_argument("--batch_size",       type=int,    help="Batch size per each GPU",
                    default=32)
parser.add_argument("--epoch",            type=int,    help="Training epochs",
                    default=100)
parser.add_argument("--lr",               type=float,  help="Initial learning rate",
                    default=0.0004)
parser.add_argument("--weight_decay",     type=float,  help="Set Weight Decay",
                    default=0.00001)
parser.add_argument("--image_size",       type=tuple,  help="Set model input size",
                    default=(480, 640))
parser.add_argument("--optimizer",        type=str,    help="Set optimizer",
                    default='adam')
parser.add_argument("--use_weight_decay",  type=bool,   help="Whether to use weightDecay",
                    default=False)
parser.add_argument("--mixed_precision",  type=bool,   help="Whether to use mixed_precision",
                    default=True)
parser.add_argument("--model_name",       type=str,    help="Set the model name to save",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))

# Set directory path (Dataset,  Dataset_type, Chekcpoints, Tensorboard)
parser.add_argument("--dataset_dir",      type=str,    help="Set the dataset download directory",
                    default='./datasets/')
parser.add_argument("--dataset_name",     type=str,    help="Set the dataset type",
                    default='nyu_depth_v2')
parser.add_argument("--checkpoint_dir",   type=str,    help="Set the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,    help="Set tensorboard storage path",
                    default='tensorboard/')

# Set Distribute training (When use Single gpu)
parser.add_argument("--gpu_num",          type=int,    help="Set GPU number to use(When without distribute training)",
                    default=1)

# Set Distribute training (When use Multi gpu)
parser.add_argument("--multi_gpu",  help="Set up distributed learning mode", action='store_true')

args = parser.parse_args()


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()
    if args.multi_gpu == False:
        tf.config.set_soft_device_placement(True)

        gpu_number = '/device:GPU:' + str(args.gpu_num)
        with tf.device(gpu_number):
            model = ModelConfiguration(args=args)
            model.train()

    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = ModelConfiguration(args=args, mirrored_strategy=mirrored_strategy)
            model.train()