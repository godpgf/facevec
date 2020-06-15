from tensorflow.python import pywrap_tensorflow
import argparse
import tensorflow as tf
import scipy.io as sio
import os

parser = argparse.ArgumentParser()
req_grp = parser.add_argument_group('required')
req_grp.add_argument('model_path', default=None, help='directory to load checkpoints from.')
args = parser.parse_args()

if __name__ == "__main__":
    # 首先，使用tensorflow自带的python打包库读取模型
    print("restore from " + tf.train.latest_checkpoint(args.model_path))
    model_reader = pywrap_tensorflow.NewCheckpointReader(tf.train.latest_checkpoint(args.model_path))

    # 然后，使reader变换成类似于dict形式的数据
    var_dict = model_reader.get_variable_to_shape_map()
    for key, value in var_dict.items():
        print(key)
    assert False
    layers = (
        "sphere/conv1_/conv1/conv2d/kernel",
        "sphere/conv1_/conv1/conv2d/bias",
        "sphere/conv1_/conv1/conv1/alpha",
        "sphere/conv1_/conv1_23/conv2d/kernel",
        "sphere/conv1_/conv1_23/name1/alpha",
        "sphere/conv1_/conv1_23/conv2d_1/kernel",
        "sphere/conv1_/conv1_23/name2/alpha",

        "sphere/conv2_/conv2/conv2d/kernel",
        "sphere/conv2_/conv2/conv2d/bias",
        "sphere/conv2_/conv2/conv2/alpha",
        "sphere/conv2_/conv2_23/conv2d/kernel",
        "sphere/conv2_/conv2_23/name1/alpha",
        "sphere/conv2_/conv2_23/conv2d_1/kernel",
        "sphere/conv2_/conv2_23/name2/alpha",
        "sphere/conv2_/conv2_45/conv2d/kernel",
        "sphere/conv2_/conv2_45/name1/alpha",
        "sphere/conv2_/conv2_45/conv2d_1/kernel",
        "sphere/conv2_/conv2_45/name2/alpha",

        "sphere/conv3_/conv3/conv2d/kernel",
        "sphere/conv3_/conv3/conv2d/bias",
        "sphere/conv3_/conv3/conv3/alpha",
        "sphere/conv3_/conv3_23/conv2d/kernel",
        "sphere/conv3_/conv3_23/name1/alpha",
        "sphere/conv3_/conv3_23/conv2d_1/kernel",
        "sphere/conv3_/conv3_23/name2/alpha",
        "sphere/conv3_/conv3_45/conv2d/kernel",
        "sphere/conv3_/conv3_45/name1/alpha",
        "sphere/conv3_/conv3_45/conv2d_1/kernel",
        "sphere/conv3_/conv3_45/name2/alpha",
        "sphere/conv3_/conv3_67/conv2d/kernel",
        "sphere/conv3_/conv3_67/name1/alpha",
        "sphere/conv3_/conv3_67/conv2d_1/kernel",
        "sphere/conv3_/conv3_67/name2/alpha",
        "sphere/conv3_/conv3_89/conv2d/kernel",
        "sphere/conv3_/conv3_89/name1/alpha",
        "sphere/conv3_/conv3_89/conv2d_1/kernel",
        "sphere/conv3_/conv3_89/name2/alpha",

        "sphere/conv4_/conv4/conv2d/kernel",
        "sphere/conv4_/conv4/conv2d/bias",
        "sphere/conv4_/conv4/conv4/alpha",
        "sphere/conv4_/conv4_23/conv2d/kernel",
        "sphere/conv4_/conv4_23/name1/alpha",
        "sphere/conv4_/conv4_23/conv2d_1/kernel",
        "sphere/conv4_/conv4_23/name2/alpha",

        "sphere/feature/dense/kernel",
        "sphere/feature/dense/bias"
    )

    # face identity encoder
    fie = {}
    for layer in layers:
        assert layer in var_dict
        fie[layer] = model_reader.get_tensor(layer)

    sio.savemat(os.path.join(args.model_path, "fie.mat"), fie)
    # print(sio.loadmat(os.path.join(args.model_path, "fie.mat"))[layers[2]])
