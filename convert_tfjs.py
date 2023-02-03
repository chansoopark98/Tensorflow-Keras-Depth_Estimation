import tensorflow as tf
from model.model_builder import ModelBuilder
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import argparse
import os 

# tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16
# tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_uint8 '*'
# tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity'

# tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='Identity' ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/

# keras convert
# tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./checkpoints/ ./checkpoints/converted_tfjs/ --quantize_float16 --control_flow_v2=True  --strip_debug_ops=True


# test
# tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity'
# tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16 '*'  --control_flow_v2=True
# tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16 '*'  --control_flow_v2=True  --strip_debug_ops=True
# tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_uint8 '*' --control_flow_v2=True  --strip_debug_ops=True

# optimized pb
# python -m tensorflow.python.tools.optimize_for_inference --input ./checkpoints/converted_frozen_graph/frozen_graph.pb --output ./checkpoints/optimized/optmized_graph.pb --frozen_graph=True --input_names=x --output_names=Identity
# tensorflowjs_converter ./checkpoints/optimized/optmized_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_uint8 '*' --control_flow_v2=True  --strip_debug_ops=True
# tensorflowjs_converter ./checkpoints/optimized/optmized_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16 '*' --control_flow_v2=True  --strip_debug_ops=True


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir",      type=str,    help="Set the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--model_weights",       type=str,    help="Saved model weights directory",
                    default='0203/_Bs-32_Ep-30_Lr-0.0002_ImSize-256_Opt-adamW_multi-gpu_0203_MobileDepth_scale0to10_multi_16:9_out128x64_best_ssim.h5')
parser.add_argument("--image_size",          type=tuple,  help="Set image size for priors and post-processing",
                    default=(256, 128))
parser.add_argument("--gpu_num",             type=int,    help="Set GPU number to use(When without distribute training)",
                    default=0)
parser.add_argument("--frozen_dir",          type=str,    help="Path to save frozen graph transformation result",
                    default='./checkpoints/converted_frozen_graph/')
parser.add_argument("--frozen_name",         type=str,    help="Frozen graph file name to save",
                    default='frozen_graph')
parser.add_argument("--include_postprocess",   help="Frozen graph file name to save",
                    action='store_true')
            
args = parser.parse_args()

if __name__ == '__main__':
    tf.config.set_soft_device_placement(True)
    # tf.config.run_functions_eagerly(True)
    # from tensorflow.python.framework.ops import disable_eager_execution
    # disable_eager_execution()
    gpu_number = '/device:GPU:' + str(args.gpu_num)
    with tf.device(gpu_number):

    
        model_builder = ModelBuilder(image_size=args.image_size,
                                  use_weight_decay=False,
                                  weight_decay=0,
                                  is_tunning=False)

        # Build model by model name
        model = model_builder.build_model()
        model.load_weights(args.checkpoint_dir + args.model_weights)
        model.summary()

        model_input = model.input

        model_output = tf.divide(model.output[0], 10.)
        model_output = tf.clip_by_value(model_output, 0., 1.)

        model = tf.keras.models.Model(model_input, model_output)

        frozen_out_path = args.frozen_dir

        os.makedirs(frozen_out_path, exist_ok=True)
        
        frozen_graph_filename = args.frozen_name
        
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        
        # print("Frozen model layers: ")
        # for layer in layers:
        #     print(layer)
        
        print("Frozen model inputs: {0}".format(frozen_func.inputs))
        print("Frozen model outputs: {0}".format(frozen_func.outputs))
        
        # Save frozen graph to disk
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pb",
                        as_text=False)
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pbtxt",
                        as_text=True)