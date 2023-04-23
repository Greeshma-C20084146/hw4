import tensorflow.compat.v1 as tf
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import os

batch_size=64
learning_rate= 5e-5
num_epochs=50000

(train_X, train_y), (_, _) = cifar10.load_data()
print("20")
print("- Training-set_image:\t\t{}".format(np.shape(train_X)))

data = []

for img in train_X:
    image = np.transpose(np.reshape(img, (32, 32, 3), 'F'), (1, 0, 2))
    image = image/255
    image = image * 2 - 1
    data.append(image)

"""#Generator"""

def generator(g, training=True,reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        g = tf.layers.dense(g, units= 4 * 4 * 512, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.reshape(g, shape=[-1, 4, 4, 512])
        g = tf.layers.batch_normalization(g)
        g = tf.nn.relu(g)
       
        g = tf.layers.conv2d_transpose(g, 256, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.layers.batch_normalization(g, momentum=0.99, training=training)
        g = tf.nn.relu(g)
       
        g = tf.layers.conv2d_transpose(g, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.layers.batch_normalization(g, momentum=0.99, training=training)
        g = tf.nn.relu(g)
       
        g = tf.layers.conv2d_transpose(g, 64, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.layers.batch_normalization(g, momentum=0.99, training=training)
        g = tf.nn.relu(g)
       
        g = tf.layers.conv2d_transpose(g, 3, 5, strides=1, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        image = tf.nn.tanh(g)
        
        return image

"""#Discriminator"""

def discriminator(d, training=True, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        d = tf.layers.conv2d(d, 64, 5, strides=2, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.nn.leaky_relu(d)
        d = tf.layers.conv2d(d, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.layers.batch_normalization(d, momentum=0.99, training=training)
        d = tf.nn.leaky_relu(d)
        d = tf.layers.conv2d(d, 256, 5, strides=2, padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.layers.batch_normalization(d, momentum=0.99, training=training)
        d = tf.nn.leaky_relu(d)    
        d = tf.layers.conv2d(d, 512, 1, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.layers.batch_normalization(d, momentum=0.99, training=training)
        d = tf.nn.leaky_relu(d)

        d = tf.layers.flatten(d)
        d = tf.layers.dense(d, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        return d

tf.disable_eager_execution()

tf.reset_default_graph()

noise_input = tf.placeholder(dtype=tf.float32,shape= [None, 100], name="NoiseInputGenerator")

real_Image_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="ImageInputDiscriminator")

generated_image=generator(noise_input)
d_real_logit=discriminator(real_Image_input)
d_fake_logit=discriminator(generated_image,reuse=True)

lr = tf.placeholder(tf.float32, shape = [], name = 'lr')
training_mode = tf.placeholder(tf.bool)

"""#Loss Calculations"""

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logit, labels=tf.ones_like(d_real_logit)))

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit, labels=tf.zeros_like(d_fake_logit)))

discriminator_loss = d_loss_real + d_loss_fake

generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit, labels=tf.ones_like(d_fake_logit)))

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

g_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Generator')
d_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Discriminator')

with tf.control_dependencies(g_ops):
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(generator_loss,var_list=g_vars)
with tf.control_dependencies(d_ops):
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(discriminator_loss, var_list=d_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

try:
    # Training
    idx = 0
    total_data = len(data)
    # batch_size = 50
    fake_acc = 0
    # learning_rate = 5e-5
    for i in range(num_epochs):
        for j in range(3):
            noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)
            if (idx + batch_size >= total_data):
                idx = 0
            examples_image = data[idx : idx + batch_size]
            idx += batch_size
            _, loss_disc = sess.run([discriminator_optimizer, discriminator_loss], 
                                        feed_dict={real_Image_input: examples_image, noise_input: noise, 
                                                   training_mode:True, lr: learning_rate})
        noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)
        if (idx + batch_size >= total_data):
            idx = 0
        examples_image = data[idx : idx + batch_size]
        idx += batch_size
            
        _, loss_gen = sess.run([generator_optimizer,generator_loss], 
                               feed_dict={noise_input: noise,
                                         training_mode:True, lr: learning_rate})
        fake_acc += sess.run(tf.reduce_mean(tf.round(tf.nn.sigmoid(d_fake_logit))), {noise_input: noise})
        saver.save(sess, "./model_DCGAN/model_backup.ckpt")
        if i % 100==0 or i==1:
            print("Epoch: {}, Generator Loss:{}, Discriminator Loss: {}, Fake Acc: {}".format(i,loss_gen, loss_disc, fake_acc/100))
            saver.save(sess, f"./model_DCGAN/model_{i}.ckpt")
            fake_acc = 0
except KeyboardInterrupt:
    pass

# def save_pics(pic_list):
#     i = 0
#     for pic in pic_list:
#         i += 1
#         sizes = np.shape(pic)     
#         fig = plt.figure()
#         fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
#         ax = plt.Axes(fig, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         fig.add_axes(ax)
#         if not os.path.exists('generated_images'):
#             os.makedirs('generated_images')
#         plt.savefig(f"generated_images/DCGAN_pic_{i}", dpi = sizes[0]) 
#         plt.close()

imgs = sess.run(generated_image, 
                           feed_dict={noise_input: np.random.uniform(-1, 1, size=[64, 100]).astype(np.float32)})
imgs = (imgs + 1)/2
# save_pics(imgs)

for i,img in enumerate(imgs):
    plt.imshow(img)
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    plt.imsave(f"generated_images/DCGAN_pic_{i}.png",img)

# plt.figure(figsize=(18, 18))
# for i in range(12):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(imgs[i])

# folder_path = '/content/generated_images'
# images = os.listdir(folder_path)

# from PIL import Image

# plt.figure(figsize=(18, 18))
# for i ,image_name in enumerate(images[:11]):
#     image_path = os.path.join(folder_path,image_name)
#     img = Image.open(image_path)
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(img)

