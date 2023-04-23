
import tensorflow.compat.v1 as tf
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import os

batch_size=64
learning_rate= 1e-3
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

"""#Generator  

"""

def generator(g, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        g = tf.layers.dense(g, units=4 * 4 * 512, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.reshape(g, shape=[-1, 4, 4, 512])
        g = tf.nn.relu(g)
        g = tf.layers.conv2d_transpose(g, 256, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.nn.relu(g)
        print(g)
        g = tf.layers.conv2d_transpose(g, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.nn.relu(g)
        print(g)
        g = tf.layers.conv2d_transpose(g, 64, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.nn.relu(g)
        print(g)
        g = tf.layers.conv2d_transpose(g, 3, 5, strides=1, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        print(g)
        image = tf.nn.tanh(g)
        return image

"""#Discriminator 

"""

def discriminator(d, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        d = tf.layers.conv2d(d, 64, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.nn.leaky_relu(d)
        print(d)
        d = tf.layers.conv2d(d, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.nn.leaky_relu(d)
        print(d)
        d = tf.layers.conv2d(d, 256, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.nn.leaky_relu(d)    
        print(d)
        d = tf.layers.flatten(d)
        d = tf.layers.dense(d, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        return d

tf.disable_eager_execution()
tf.reset_default_graph()
generator_lr = tf.placeholder(tf.float32, shape=[])
critic_lr = tf.placeholder(tf.float32, shape=[])

noise_input = tf.placeholder(dtype=tf.float32,shape= [None, 100], name="NoiseInputGenerator")

real_Image_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="ImageInputDiscriminator")

generated_image=generator(noise_input)
d_real_logit=discriminator(real_Image_input)
d_fake_logit=discriminator(generated_image,reuse=True)

"""#Loss Calculations

"""

d_loss_real = tf.reduce_mean(d_real_logit)

d_loss_fake = tf.reduce_mean(d_fake_logit)

discriminator_loss = (d_loss_real - d_loss_fake)

generator_loss = - d_loss_fake

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

discriminator_clip=[c.assign(tf.clip_by_value(c,-0.01,0.01)) for c in d_vars]

generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=generator_lr).minimize(generator_loss,var_list=g_vars)
discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=critic_lr).minimize(-discriminator_loss, var_list=d_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

try:
    # Training
    fake_acc = 0
    glr = 5e-4
    clr = 5e-4
    avg_disc_loss = 0
    avg_gen_loss = 0
    intervals = 100
    total_data = len(data)
    idx = 0
    for i in range(num_epochs):
        for critic in range(6):
            if (idx + batch_size >= total_data):
                idx = 0
            examples_image = data[idx : idx + batch_size]
            idx += batch_size
            noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)
        
            # Discriminator Training
            _, loss_disc = sess.run([discriminator_optimizer, discriminator_loss], 
                                feed_dict={real_Image_input: examples_image, noise_input: noise, critic_lr: clr})
            sess.run(discriminator_clip)
        avg_disc_loss += loss_disc
        if (idx + batch_size >= total_data):
            idx = 0
        examples_image = data[idx : idx + batch_size]
        idx += batch_size
        noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)


        _, loss_gen = sess.run([generator_optimizer,generator_loss], 
                                   feed_dict={noise_input: noise, generator_lr: glr})
        avg_gen_loss += loss_gen
        saver.save(sess, "./model_WCGAN/model_backup.ckpt")
        if i % intervals == 0 or i == 1:
            print("Epoch: {}, Generator Loss:{}, Discriminator Loss: {}".format(i,avg_gen_loss/intervals, avg_disc_loss/intervals))
            saver.save(sess, f"./model_WCGAN/model_{i}.ckpt")
            avg_gen_loss = 0
            avg_disc_loss = 0
except KeyboardInterrupt:
    pass

"""#Generate Images"""

imgs = sess.run(generated_image, 
                           feed_dict={noise_input: np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)})
imgs = (imgs + 1)/2
# save_pics(imgs)

for i,img in enumerate(imgs):
    plt.imshow(img)
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    plt.imsave(f"generated_images/WGAN_pic_{i}.png",img)
