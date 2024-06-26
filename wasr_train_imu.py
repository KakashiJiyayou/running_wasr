import argparse
import os
import tensorflow as tf
import numpy as np

from wasr_models import wasr_IMU_FU2, decode_labels, inv_preprocess, prepare_label

# COLOR MEANS OF IMAGES FROM MODDv1 DATASET
IMG_MEAN = np.array((148.8430, 171.0260, 162.4082), dtype=np.float32)

BATCH_SIZE = 2
DATA_DIRECTORY = '/opt/workspace/host_storage_hdd/boat/train_images_mastr_all/'
DATA_LIST_PATH = '/opt/workspace/host_storage_hdd/boat/train_images_mastr_all/train_water_deformed.txt'

GRAD_UPDATE_EVERY = 10
IGNORE_LABEL = 4
INPUT_SIZE = '384,512'
LEARNING_RATE = 1e-6
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 80001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = '/opt/workspace/host_storage_hdd/boat/weights_models/snapshots_wasr_imu_fu2/'
WEIGHT_DECAY = 1e-6

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--grad-update-every", type=int, default=GRAD_UPDATE_EVERY,
                        help="Number of steps after which gradient update is applied.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to update the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def focal_loss_cost(labels, logits, gamma=2.0, alpha=4.0):
    epsilon = 1.e-9
    softmax_logits = tf.add(tf.nn.softmax(logits), epsilon)
    mask_o = tf.cast(tf.equal(labels, 0), dtype=tf.float32)
    mask_w = tf.cast(tf.equal(labels, 1), dtype=tf.float32)
    mask_s = tf.cast(tf.equal(labels, 2), dtype=tf.float32)

    fl_ce_o = -1. * mask_o * tf.math.log(softmax_logits[:,0]) * (1. - softmax_logits[:,0]) ** gamma
    fl_ce_w = -1. * mask_w * tf.math.log(softmax_logits[:,1]) * (1. - softmax_logits[:,1]) ** gamma
    fl_ce_s = -1. * mask_s * tf.math.log(softmax_logits[:,2]) * (1. - softmax_logits[:,2]) ** gamma

    fl_ce = fl_ce_o + fl_ce_w + fl_ce_s
    return tf.reduce_mean(fl_ce)

def cost_function_separate_water_obstacle(features_output, gt_mask):
    epsilon_watercost = 0.01
    features_shape = features_output.get_shape()
    gt_mask = tf.image.resize(gt_mask, size=features_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    mask_water = tf.equal(gt_mask[:, :, :, 0], 1)
    mask_water = tf.expand_dims(mask_water, 3)
    mask_water = tf.cast(mask_water, dtype=tf.float32)

    mask_obstacles = tf.equal(gt_mask[:, :, :, 0], 0)
    mask_obstacles = tf.expand_dims(mask_obstacles, 3)
    mask_obstacles = tf.cast(mask_obstacles, dtype=tf.float32)

    elements_water = tf.reduce_sum(mask_water, axis=[1, 2])
    elements_obstacles = tf.reduce_sum(mask_obstacles, axis=[1, 2])

    elements_obstacles = tf.where(tf.equal(elements_obstacles, 0), tf.ones_like(elements_obstacles), elements_obstacles)
    elements_water = tf.where(tf.equal(elements_water, 0), tf.ones_like(elements_water), elements_water)

    water_pixels = features_output * mask_water
    obstacle_pixels = features_output * mask_obstacles

    mean_water = tf.reduce_mean(tf.divide(tf.reduce_sum(water_pixels, axis=[1, 2]), elements_water), axis=0, keepdims=True)
    mean_water_matrix = tf.expand_dims(mean_water, 1)
    mean_water_matrix_all = tf.expand_dims(mean_water_matrix, 1)
    mean_water_matrix_wat = mean_water_matrix_all * mask_water
    mean_water_matrix_obs = mean_water_matrix_all * mask_obstacles

    var_water = tf.divide(tf.reduce_sum(tf.math.squared_difference(water_pixels, mean_water_matrix_wat), axis=[1, 2]), elements_water)
    var_water = tf.reduce_mean(var_water, axis=0, keepdims=True)

    difference_obs_wat = tf.reduce_sum(tf.math.squared_difference(obstacle_pixels, mean_water_matrix_obs), axis=[1, 2])
    loss_c = tf.divide(var_water + epsilon_watercost, tf.divide(difference_obs_wat, elements_obstacles) + epsilon_watercost)
    var_cost = tf.reduce_mean(loss_c)

    return var_cost

def create_dataset(data_list_path, input_size, batch_size, img_mean):
    def _parse_function(filename, label, imu):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, input_size)
        image = tf.cast(image, tf.float32) - img_mean

        label_string = tf.io.read_file(label)
        label = tf.image.decode_png(label_string, channels=1)
        label = tf.image.resize(label, input_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        imu_string = tf.io.read_file(imu)
        imu = tf.image.decode_png(imu_string, channels=1)
        imu = tf.image.resize(imu, input_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image, label, imu

    dataset = tf.data.TextLineDataset(data_list_path)
    dataset = dataset.map(lambda line: tf.strings.split(line))
    dataset = dataset.map(lambda parts: (parts[0], parts[1], parts[2]))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat()

    return dataset

class WASRModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(WASRModel, self).__init__()
        self.model = wasr_IMU_FU2({'data': None, 'imu_data': None}, is_training=True, num_classes=num_classes)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def call(self, inputs, training=False):
        data, imu_data = inputs
        return self.model({'data': data, 'imu_data': imu_data}, training=training)

    def train_step(self, data):
        image_batch, label_batch, imu_batch = data
        with tf.GradientTape() as tape:
            outputs = self((image_batch, imu_batch), training=True)
            raw_output = outputs['fc1_voc12']
            inthemiddle_output = outputs['res4b20']

            raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
            label_proc = prepare_label(label_batch, tf.shape(raw_output)[1:3], num_classes=args.num_classes, one_hot=False)
            raw_gt = tf.reshape(label_proc, [-1,])

            indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
            gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
            prediction = tf.gather(raw_prediction, indices)

            loss_0 = cost_function_separate_water_obstacle(inthemiddle_output, label_batch)
            l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in self.trainable_variables if 'weights' in v.name]
            added_l2_losses = 10.e-2 * tf.add_n(l2_losses)
            focal_loss = focal_loss_cost(labels=gt, logits=prediction)
            total_loss = added_l2_losses + focal_loss + loss_0

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    tf.random.set_seed(args.random_seed)

    dataset = create_dataset(args.data_list, input_size, args.batch_size, IMG_MEAN)

    model = WASRModel(num_classes=args.num_classes)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate, momentum=args.momentum))

    steps_per_epoch = args.num_steps // args.batch_size
    model.fit(dataset, epochs=args.num_steps // steps_per_epoch, steps_per_epoch=steps_per_epoch)

    # Save the model at the end of training
    model.save(os.path.join(args.snapshot_dir, 'final_model.h5'))

if __name__ == '__main__':
    main()
