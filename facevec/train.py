from data_pipeline import ImageFileDataPipeline
from models import Sphere
import argparse
import tensorflow as tf
import logging
import os

logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Sphere Face')
parser.add_argument('checkpoint_dir', default=None, help='directory to save checkpoints to.')
parser.add_argument('file_list', default=None, help='input file_list.')

parser.add_argument('--embedding_size', default=512, type=int, help='size of feature embedding.')

# train
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='learning rate for the stochastic gradient update.')
parser.add_argument('--log_interval', type=int, default=1, help='interval between log messages (in s).')
parser.add_argument('--summary_interval', type=int, default=120,
                    help='interval between tensorboard summaries (in s)')
parser.add_argument('--checkpoint_interval', type=int, default=600, help='interval between model checkpoints (in s)')
parser.add_argument('--height', type=int, default=112, help='image height')
parser.add_argument('--width', type=int, default=96, help='image width')

# Data pipeline and data augmentation
data_grp = parser.add_argument_group('data pipeline')
data_grp.add_argument('--batch_size', default=256, type=int, help='size of a batch for each gradient update.')
data_grp.add_argument('--data_threads', default=4, help='number of threads to load and enqueue samples.')
data_grp.add_argument('--rotate', dest="rotate", action="store_true", help='rotate data augmentation.')
data_grp.add_argument('--norotate', dest="rotate", action="store_false")
data_grp.add_argument('--flipud', dest="flipud", action="store_true", help='flip up/down data augmentation.')
data_grp.add_argument('--noflipud', dest="flipud", action="store_false")
data_grp.add_argument('--fliplr', dest="fliplr", action="store_true", help='flip left/right data augmentation.')
data_grp.add_argument('--nofliplr', dest="fliplr", action="store_false")
data_grp.add_argument('--brightness', dest="brightness", action="store_true", help='random bright data augmentation.')
data_grp.add_argument('--nobrightness', dest="brightness", action="store_false")
data_grp.add_argument('--contrast', dest="contrast", action="store_true", help='random contrast data augmentation.')
data_grp.add_argument('--nocontrast', dest="contrast", action="store_false")
data_grp.add_argument('--hue', dest="hue", action="store_true", help='random hue data augmentation.')
data_grp.add_argument('--nohue', dest="hue", action="store_false")
data_grp.add_argument('--satu', dest="satu", action="store_true", help='random satu data augmentation.')
data_grp.add_argument('--nosatu', dest="satu", action="store_false")
data_grp.add_argument('--random_crop', dest="random_crop", action="store_true", help='random crop data augmentation.')

parser.set_defaults(
    flipud=False,
    fliplr=True,
    rotate=False,
    brightness=False,
    contrast=False,
    hue=False,
    satu=False,
    random_crop=True)


# 创建优化器
def create_optimizer(loss, global_step):
    with tf.name_scope('optimizer'):
        # 关于tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用。
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        updates = tf.group(*update_ops, name='update_ops')
        log.info("Adding {} update ops".format(len(update_ops)))

        with tf.control_dependencies([updates]):
            opt = tf.train.AdamOptimizer(args.learning_rate)
            minimize = opt.minimize(loss, name='optimizer', global_step=global_step)
    return minimize


def create_log_ma(loss):
    # 创建用于日志输出的平均数
    with tf.name_scope("moving_averages"):
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        # 更新平均数
        update_ma = ema.apply([loss])
        # 读取平均数
        loss = ema.average(loss)
    return update_ma, loss


def create_log_hook(global_step, loss):
    # Save a few graphs to tensorboard
    summaries = [
        tf.summary.scalar('loss', loss),
        tf.summary.scalar('learning_rate', args.learning_rate),
        tf.summary.scalar('batch_size', args.batch_size),
    ]

    tensors_to_log = {"step": global_step.name.split(":")[0],
                      "loss": loss.name.split(":")[0]}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_secs=args.log_interval)
    return logging_hook


def get_tf_sess_factory(global_step, loss):
    # Train config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU

    logging_hook = create_log_hook(global_step, loss)

    def create_sess():
        return tf.train.MonitoredTrainingSession(
            checkpoint_dir=args.checkpoint_dir,
            save_summaries_secs=args.summary_interval,
            save_checkpoint_secs=args.checkpoint_interval,
            config=config,
            hooks=[logging_hook])

    return create_sess


if __name__ == '__main__':
    args = parser.parse_args()
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        data_pipeline = ImageFileDataPipeline(args.file_list, args.height, args.width, args.random_crop,
                                              batch_size=args.batch_size, fliplr=args.fliplr,
                                              flipud=args.flipud, rotate=args.rotate, brightness=args.brightness,
                                              contrast=args.contrast, hue=args.hue, satu=args.satu,
                                              nthreads=args.data_threads)

        feature = Sphere.inference(data_pipeline.samples["image_input"], args.embedding_size)
        # feature, _ = mobilenet_v3_large(tf.image.resize_images(data_pipeline.samples["image_input"], (128, 128)), args.embedding_size)
        logits, loss = Sphere.cos_loss(feature, data_pipeline.samples["image_label"],
                                                    data_pipeline.samples["num_cls"], args.batch_size)

        minimize = create_optimizer(loss, global_step)
        update_ma, loss = create_log_ma(loss)
        train_op = tf.group(minimize, update_ma)

        saver = tf.train.Saver(tf.global_variables())
        sess_factory = get_tf_sess_factory(global_step, loss)
        with sess_factory() as sess:
            while True:
                if sess.should_stop():
                    log.info("stopping supervisor")
                    break
                try:
                    step, _ = sess.run([global_step, train_op])
                except tf.errors.AbortedError:
                    log.error("Aborted")
                    break
                except KeyboardInterrupt:
                    break
            chkpt_path = os.path.join(args.checkpoint_dir, 'on_stop.ckpt')
            log.info("Training complete, saving chkpt {}".format(chkpt_path))


            def get_session(session):
                while type(session).__name__ != 'Session':
                    session = session._sess
                return session


            saver.save(get_session(sess), chkpt_path)
