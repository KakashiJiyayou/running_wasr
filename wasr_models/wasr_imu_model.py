import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Concatenate

class WASRIMUFU2:
    def __init__(self, input_shape, imu_shape, num_classes):
        self.input_shape = input_shape
        self.imu_shape = imu_shape
        self.num_classes = num_classes

    def conv_block(self, x, filters, kernel_size, strides, name, activation=True):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=name)(x)
        x = BatchNormalization(name=name+'_bn')(x)
        if activation:
            x = ReLU(name=name+'_relu')(x)
        return x

    def residual_block(self, input_tensor, filters, strides, block_name):
        x = self.conv_block(input_tensor, filters[0], (1, 1), strides, name=block_name + '_branch2a')
        x = self.conv_block(x, filters[1], (3, 3), (1, 1), name=block_name + '_branch2b')
        x = self.conv_block(x, filters[2], (1, 1), (1, 1), name=block_name + '_branch2c', activation=False)
        shortcut = self.conv_block(input_tensor, filters[2], (1, 1), strides, name=block_name + '_branch1', activation=False)
        x = Add(name=block_name)([x, shortcut])
        x = ReLU(name=block_name+'_relu')(x)
        return x

    def atrous_residual_block(self, input_tensor, filters, dilation_rate, block_name):
        x = self.conv_block(input_tensor, filters[0], (1, 1), (1, 1), name=block_name + '_branch2a')
        x = Conv2D(filters[1], (3, 3), strides=(1, 1), dilation_rate=dilation_rate, padding='same', use_bias=False, name=block_name + '_branch2b')(x)
        x = BatchNormalization(name=block_name + '_branch2b_bn')(x)
        x = ReLU(name=block_name + '_branch2b_relu')(x)
        x = self.conv_block(x, filters[2], (1, 1), (1, 1), name=block_name + '_branch2c', activation=False)
        shortcut = self.conv_block(input_tensor, filters[2], (1, 1), (1, 1), name=block_name + '_branch1', activation=False)
        x = Add(name=block_name)([x, shortcut])
        x = ReLU(name=block_name+'_relu')(x)
        return x

    def resize_img(self, x, size, name):
        return tf.image.resize(x, size, name=name)

    def attention_refinement_module(self, x, name, last_arm=False):
        num_channels = x.shape[-1]
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, num_channels))(x)
        x = Conv2D(num_channels, (1, 1), use_bias=False, name=name + '_conv')(x)
        x = BatchNormalization(name=name + '_bn')(x)
        x = tf.sigmoid(x)
        if last_arm:
            x = layers.Multiply()([x, x])
        return x

    def feature_fusion_module(self, x1, x2, name, num_features):
        x = Concatenate(axis=-1, name=name + '_concat')([x1, x2])
        x = Conv2D(num_features, (1, 1), use_bias=False, name=name + '_conv')(x)
        x = BatchNormalization(name=name + '_bn')(x)
        x = ReLU(name=name + '_relu')(x)
        return x

    def build_model(self, is_training):
        inputs = tf.keras.Input(shape=self.input_shape, name='data')
        imu_inputs = tf.keras.Input(shape=self.imu_shape, name='imu_data')

        # Block 1
        x = self.conv_block(inputs, 64, (7, 7), strides=(2, 2), name='conv1')
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        # Residual Block 2a
        x = self.residual_block(x, [64, 64, 256], strides=(1, 1), block_name='res2a')

        # Residual Block 2b and 2c
        x = self.residual_block(x, [64, 64, 256], strides=(1, 1), block_name='res2b')
        x = self.residual_block(x, [64, 64, 256], strides=(1, 1), block_name='res2c')

        # Resizing IMU data
        resized_imu_1 = self.resize_img(imu_inputs, (48, 64), name='resized_imu_1')
        resized_imu_0 = self.resize_img(imu_inputs, (96, 128), name='resized_imu_0')

        # Residual Block 3a
        x = self.residual_block(x, [128, 128, 512], strides=(2, 2), block_name='res3a')
        
        # Residual Block 3b1, 3b2, and 3b3
        x = self.residual_block(x, [128, 128, 512], strides=(1, 1), block_name='res3b1')
        x = self.residual_block(x, [128, 128, 512], strides=(1, 1), block_name='res3b2')
        x = self.residual_block(x, [128, 128, 512], strides=(1, 1), block_name='res3b3')

        # Atrous Residual Block 4a
        x = self.atrous_residual_block(x, [256, 256, 1024], dilation_rate=2, block_name='res4a')

        # Atrous Residual Blocks 4b1 to 4b22
        for i in range(1, 23):
            x = self.atrous_residual_block(x, [256, 256, 1024], dilation_rate=2, block_name=f'res4b{i}')

        # Atrous Residual Block 5a
        x = self.atrous_residual_block(x, [512, 512, 2048], dilation_rate=4, block_name='res5a')
        
        # Atrous Residual Blocks 5b and 5c
        x = self.atrous_residual_block(x, [512, 512, 2048], dilation_rate=4, block_name='res5b')
        x = self.atrous_residual_block(x, [512, 512, 2048], dilation_rate=4, block_name='res5c')

        # ARM module
        concatenated_imu_arm_1 = layers.Concatenate(axis=3, name='concatenated_imu_arm_1')([x, resized_imu_1])
        arm = self.attention_refinement_module(concatenated_imu_arm_1, name='arm')

        # Additional branches and layers
        fc1_voc12_c0_first = Conv2D(32, (3, 3), dilation_rate=6, padding='same', name='fc1_voc12_c0_first')(x)
        fc1_voc12_c1_first = Conv2D(32, (3, 3), dilation_rate=12, padding='same', name='fc1_voc12_c1_first')(x)
        fc1_voc12_c2_first = Conv2D(32, (3, 3), dilation_rate=18, padding='same', name='fc1_voc12_c2_first')(x)

        fc1_voc12_first = Add(name='fc1_voc12_first')([fc1_voc12_c0_first, fc1_voc12_c1_first, fc1_voc12_c2_first])

        ffm_first = self.feature_fusion_module(arm, fc1_voc12_first, name='ffm_first', num_features=1024)

        concatenated_imu_arm_2 = layers.Concatenate(axis=3, name='concatenated_imu_arm_2')([x, resized_imu_1])
        arm_2 = self.attention_refinement_module(concatenated_imu_arm_2, name='arm_2', last_arm=True)

        arm_2_upfeatures_new = self.conv_block(arm_2, 1024, (1, 1), strides=(1, 1), name='arm_2_upfeatures_new', activation=False)

        combine_arm1_arm2 = Add(name='combine_arm1_arm2')([ffm_first, arm_2_upfeatures_new])

        concatenated_imu_ffm_1 = layers.Concatenate(axis=3, name='concatenated_imu_ffm_1')([x, resized_imu_0])

        ffm = self.feature_fusion_module(concatenated_imu_ffm_1, combine_arm1_arm2, name='ffm', num_features=1024)

        fc1_voc12_c0 = Conv2D(self.num_classes, (3, 3), dilation_rate=6, padding='same', name='fc1_voc12_c0')(ffm)
        fc1_voc12_c1 = Conv2D(self.num_classes, (3, 3), dilation_rate=12, padding='same', name='fc1_voc12_c1')(ffm)
        fc1_voc12_c2 = Conv2D(self.num_classes, (3, 3), dilation_rate=18, padding='same', name='fc1_voc12_c2')(ffm)
        fc1_voc12_c3 = Conv2D(self.num_classes, (3, 3), dilation_rate=24, padding='same', name='fc1_voc12_c3')(ffm)

        fc1_voc12 = Add(name='fc1_voc12')([fc1_voc12_c0, fc1_voc12_c1, fc1_voc12_c2, fc1_voc12_c3])

        output_layer = fc1_voc12

        model = models.Model(inputs=[inputs, imu_inputs], outputs=output_layer)
        return model


# Usage
# input_shape = (224, 224, 3)
# imu_shape = (64, 64, 3)
# num_classes = 21
# is_training = True

# wasr_imu_model = WASRIMUFU2(input_shape, imu_shape, num_classes)
# model = wasr_imu_model.build_model(is_training)
# model.summary()
