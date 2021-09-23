import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file='resnet_Thien/eval_graph_frozen.pb',
                input_arrays=["input_image"],
                input_shapes= {"input_image" : [1,124,124,3]},
                output_arrays=['classification_output/act_quant/FakeQuantWithMinMaxVars'])
converter.quantized_input_stats = {"input_image" : (0, 1)}
converter.inference_type = tf.uint8
converter.inference_input_type = tf.uint8
quantized_tflite_model = converter.convert()
with open("test_Thien.tflite", 'wb') as f:
    f.write(quantized_tflite_model)
