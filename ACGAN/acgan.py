import tensorflow.compat.v1 as tf
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

num_epcohs = 100
lr_discrim = 0.0002
beta_1_discrim = 0.5 
lr_gen = 0.0002
beta_1_gen = 0.5
batch_size = 64
z_dim = 100
CONTINUE_TRAINING = True
start_point = 0

(train_X, train_y), (_, _) = cifar10.load_data()
print("- Training-set_image:\t\t{}".format(np.shape(train_X)))

img_size_cifar = 32
num_channels_cifar = 3 
img_size_flat_cifar = img_size_cifar * img_size_cifar*num_channels_cifar
img_shape_cifar = (img_size_cifar, img_size_cifar, num_channels_cifar)
num_classes_cifar = 10

# process the data
train_X = train_X/255
train_X = train_X*2-1
train_Y = to_categorical(train_y, num_classes_cifar)

kernel_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

"""#Generator"""

def generator(z, fake_label, reuse=False):
    
    with tf.variable_scope('generator', reuse=reuse):
        

        input_to_conv = tf.layers.dense(tf.concat([z, fake_label], axis=1), 4*4*512)
        

        layer1 = tf.reshape(input_to_conv, (-1, 4, 4, 512))
        layer1 = tf.nn.leaky_relu(layer1, alpha=0.2, name='leaky_relu1_g')
        
        
        layer2 = tf.layers.conv2d_transpose(layer1, filters=256, kernel_size=5, strides= 2, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution2')
        layer2 = tf.nn.leaky_relu(layer2, alpha=0.2, name='leaky_relu2_g')
        
 
        layer3 = tf.layers.conv2d_transpose(layer2, filters=128, kernel_size=5, strides= 2, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution3')
        
        
        layer3 = tf.nn.leaky_relu(layer3, alpha=0.2, name='leaky_relu3_g')
        

        layer4 = tf.layers.conv2d_transpose(layer3, filters=256, kernel_size=5, strides= 2, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution4')
        
        layer4 = tf.nn.leaky_relu(layer4, alpha=0.2, name='leaky_relu4_g')
        
        

        layer5 = tf.layers.conv2d_transpose(layer4, filters=3, kernel_size=5, strides=1, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution5')
           
        
        logits = tf.tanh(layer5, name='tanh')
        
        return logits

"""#Discriminator"""

def discriminator(input_images, reuse=False):
    
    with tf.variable_scope('discriminator', reuse= reuse):
        

        
        layer1 = tf.layers.conv2d(input_images, filters=64, 
                                  kernel_size=3, strides=2, 
                                  padding='same', kernel_initializer=kernel_init, name='conv1')

        layer1 = tf.nn.leaky_relu(layer1, alpha=0.2, name='leaky_relu1')
    
        
        layer2 = tf.layers.conv2d(layer1, 
                                  filters=128, 
                                  kernel_size=3, 
                                  strides=2, 
                                  padding='same', 
                                  kernel_initializer=kernel_init, 
                                  name='conv2')
        
        layer2 = tf.nn.leaky_relu(layer2, alpha=0.2, name='leaky_relu2')

        layer3 = tf.layers.conv2d(layer2, 
                                  filters=128, 
                                  kernel_size=3, 
                                  strides=2, 
                                  padding='same', 
                                  kernel_initializer=kernel_init, 
                                  name='conv3')
        
        layer3 = tf.nn.leaky_relu(layer3, alpha=0.2, name='leaky_relu3')
 


        layer4 = tf.layers.conv2d(layer3, 
                                 filters=256, 
                                 kernel_size=3, 
                                 strides=2,
                                 padding='same',
                                 name='conv4')

        layer4 = tf.nn.leaky_relu(layer4, alpha=0.2, name='leaky_relu4')
        
        
        layer4 = tf.layers.flatten(layer4)

        layer4 = tf.nn.dropout(layer4, keep_prob=0.6)
        
        logits_discrim= tf.layers.dense(layer4, 1)
        
        output_discrim = tf.sigmoid(logits_discrim)
        
        net_1 = tf.layers.dense(inputs=layer4, name='aux_fc1', units=128, activation=tf.nn.relu)
        logits_label = tf.layers.dense(inputs=net_1, name='aux_fc_out', units=num_classes_cifar, activation=None)
        output_label_one_hot = tf.nn.softmax(logits=logits_label)
        output_label_s = tf.argmax(output_label_one_hot, dimension=1)
        
        return logits_discrim, logits_label, output_label_s

tf.disable_eager_execution()

x = tf.placeholder(tf.float32, shape= (None, img_size_cifar, img_size_cifar, num_channels_cifar), name="d_input")
label_true = tf.placeholder(tf.float32, [None, num_classes_cifar], name="label_true")

z = tf.placeholder(tf.float32, shape= (None, z_dim), name="z_noise")
label_fake = tf.placeholder(tf.float32, shape= (None, num_classes_cifar), name="label_fake")

is_training = tf.placeholder(tf.bool, [], name='is_training')

fake_x=generator(z, label_fake)

D_logit_real_discrim, D_logit_real_label, D_output_real_label = discriminator(x, reuse=False)

D_logit_fake_discrim, D_logit_fake_label, D_output_real_label = discriminator(fake_x, reuse=True)

"""#Loss Calculations"""

D_loss_real_discrim = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real_discrim),logits=D_logit_real_discrim))

D_loss_real_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_true,logits=D_logit_real_label))

D_loss_fake_discrim = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake_discrim),logits=D_logit_fake_discrim))

D_loss_fake_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_fake,logits=D_logit_fake_label))


D_loss = D_loss_real_discrim + D_loss_real_label + D_loss_fake_discrim + D_loss_fake_label

G_loss_discrim = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake_discrim),logits=D_logit_fake_discrim))

G_loss = G_loss_discrim + D_loss_fake_label

training_vars = tf.trainable_variables()

theta_D = [var for var in training_vars if var.name.startswith('discriminator')]
theta_G = [var for var in training_vars if var.name.startswith('generator')]

d_optimizer = tf.train.AdamOptimizer(lr_discrim, beta_1_discrim).minimize(D_loss, var_list=theta_D)
g_optimizer = tf.train.AdamOptimizer(lr_gen, beta_1_gen).minimize(G_loss, var_list=theta_G)

def get_fake_labels(num_labels):
    fake_label_value = np.random.randint(num_classes_cifar, size=num_labels)
    fake_label_value = fake_label_value.reshape(-1,1)
    fake_label_value = to_categorical(fake_label_value, num_classes_cifar)
    return fake_label_value

num_batches = int(train_X.shape[0] / batch_size)
loss_tracker_epoch = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

try:
    for epoch in range(num_epcohs):
        
        total_index = np.arange(len(train_X))
        
        np.random.shuffle(total_index)
        train_X_1 = np.take(a=train_X, indices=total_index, axis=0)
        train_Y_1 = np.take(a=train_Y, indices=total_index, axis=0)
        
        discrim_loss_list = []
        gen_loss_list = []
        
        for i in range(batch_size):

            start = i * batch_size
            end = (i + 1) * batch_size
            
            batch_images = train_X_1[start:end]
            batch_label_real = train_Y_1[start:end]
            batch_label_fake = get_fake_labels(batch_size)
            z_noise = np.random.uniform(-1,1, size=[batch_size, z_dim]).astype(np.float32)

            if(i % 2 == 0):
                _, discrim_loss = sess.run([d_optimizer,D_loss], feed_dict={x: batch_images, label_true:batch_label_real,label_fake:batch_label_fake, z: z_noise})

            _, gen_loss = sess.run([g_optimizer,G_loss], feed_dict={x: batch_images,label_fake:batch_label_fake, z: z_noise})
            
            _, discrim_loss = sess.run([d_optimizer,D_loss], feed_dict={x: batch_images, label_true:batch_label_real,label_fake: batch_label_fake, z: z_noise})
        
            discrim_loss_list.append(discrim_loss)
            gen_loss_list.append(gen_loss)
        saver.save(sess, "./model_ACGAN/model_backup.ckpt")
        if epoch % 100==0 or epoch==1:
            print("Epoch: {},  Discriminator Loss:{}, Generator Loss: {}".format(epoch, discrim_loss, gen_loss))
            saver.save(sess, f"./model_ACGAN/model_{i}.ckpt")
                 
except KeyboardInterrupt:
    pass

"""#Generate Images"""

def get_fake_labels(num_labels):
    fake_label_value = np.random.randint(num_classes_cifar, size=num_labels)
    fake_label_value = fake_label_value.reshape(-1,1)
    fake_label_value = to_categorical(fake_label_value, num_classes_cifar)
    return fake_label_value

cls_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def get_label(test_num_list_one_hot):
    test_label_list = []
    for test_num_one_hot in test_num_list_one_hot:
        index_1 = np.argmax(test_num_one_hot)
        test_label_list.append(cls_names[index_1])
    return test_label_list

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
#         plt.savefig(f"generated_images/ACGAN_pic_{i}", dpi = sizes[0]) 
#         plt.close()

feed_dict = {z: np.random.normal(0.0,1.0,size=[batch_size, z_dim]), label_fake: get_fake_labels(batch_size)}

imgs = sess.run(fake_x, feed_dict=feed_dict)
imgs = (imgs + 1)/2
# save_pics(imgs)

plt.figure(figsize=(18, 18))

labels_description = get_label(get_fake_labels(batch_size))
for i,img in enumerate(imgs):
    plt.imshow(img)
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    plt.imsave(f"generated_images/ACGAN_pic_{i}_{labels_description[i]}.png",img)

