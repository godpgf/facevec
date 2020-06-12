import argparse
import numpy as np
import tensorflow as tf
import os
import cv2
import skimage
from models import Sphere
from models import fie


def load_img_list(path):
    img_file_list = []
    img_label_list = []
    base_dir = os.path.dirname(path)
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            img_file, img_label = line.split(' ')
            img_file_list.append(os.path.join(base_dir, img_file))
            img_label_list.append(int(img_label))
            line = f.readline()
    return img_file_list


def run_embd(sess, img_path, height, width, img_pair_ph, img_pair_embds):
    img = cv2.imread(img_path, -1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.flip(img, 2)  # OpenCV reads BGR, convert back to RGB.
    img = skimage.img_as_float(img)
    h, w, c = img.shape
    sh = (h - height) // 2
    sw = (w - width) // 2
    img = img[sh:sh + height, sw:sw + width, :]
    f_img = img[:, ::-1, :]
    pair_imgs = np.concatenate((img[np.newaxis, :], f_img[np.newaxis, :]), axis=0)
    embds = sess.run(img_pair_embds, {img_pair_ph: pair_imgs})
    embd = embds[:1] / np.linalg.norm(embds[:1], axis=1, keepdims=True) + embds[1:2] / np.linalg.norm(embds[1:2], axis=1, keepdims=True)
    return embd[0]


parser = argparse.ArgumentParser(description='Sphere Face')
parser.add_argument('checkpoint_dir', default=None, help='directory to save checkpoints to.')
parser.add_argument('file_list', default=None, help='input file_list.')

parser.add_argument('--embedding_size', default=512, type=int, help='size of feature embedding.')
parser.add_argument('--height', type=int, default=112, help='image height')
parser.add_argument('--width', type=int, default=96, help='image width')
parser.add_argument('--target_far', type=float, default=1e-3, help='target far when calculate tar')

if __name__ == '__main__':
    args = parser.parse_args()
    img_list = load_img_list(args.file_list)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            img_pair_ph = tf.placeholder(tf.float32, [2, args.height, args.width, 3])
            if args.checkpoint_dir.endswith(".mat"):
                img_pair_embds = fie.net(args.checkpoint_dir, img_pair_ph)
            else:
                img_pair_embds = Sphere.inference(img_pair_ph, args.embedding_size)
                # 部分加载模型
                variables = tf.contrib.framework.get_variables_to_restore()
                saver = tf.train.Saver(variables)
                checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
                saver.restore(sess, checkpoint_path)
            embeds = np.reshape(np.concatenate(
                [run_embd(sess, path, args.height, args.width, img_pair_ph, img_pair_embds) for path in img_list],
                axis=0), [-1, args.embedding_size])
            np.save(os.path.join(os.path.dirname(args.file_list), "embeds.npy"), embeds)
