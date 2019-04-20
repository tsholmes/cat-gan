from keras.layers import \
    Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, \
    BatchNormalization, UpSampling2D, MaxPooling2D, \
    SpatialDropout2D, Lambda, Concatenate, Activation, \
    Add, Dot, Multiply, RepeatVector, Dropout
from keras.activations import relu
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.models import Model, Sequential
from keras import backend as K
import random
import glob
import wandb
from PIL import Image
import numpy as np
import numpy.random as nprandom
from tqdm import tqdm

run = wandb.init()
config = run.config

config.num_epochs = 1000
config.steps_per_epoch = 25
config.batch_size = 32
config.img_glob = './cat_crop/*.jpg'
config.width = 64
config.height = 64
config.gen_learning_rate = 0.0001
config.dis_learning_rate = 0.0001
config.gen_dropout = 0.4
config.dis_dropout = 0.4
config.latent_size = 100
config.examples = 20
config.batchnorm = True
config.batchnorm_momentum = 0.9
config.relu_alpha = 0.1
config.flip_rate = 0.05
config.gen_clipvalue = 0.5


image_filenames = glob.glob(config.img_glob)

def image_generator(batch_size, imgs):
    counter = 0
    random.shuffle(imgs)
    while True:
        color_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(imgs) 
        if ((counter+1)*batch_size>=len(imgs)):
            counter = 0
            random.shuffle(imgs)
        for i in range(batch_size):
            img = Image.open(imgs[counter + i]).resize((config.height, config.width))
            color_images[i] = np.array(img) / 127.5 - 1.
        yield color_images
        counter += batch_size

def calc_gradients(model, X_train, y_train):
    weights = model._collected_trainable_weights  # weight tensors

    gradients = model.optimizer.get_gradients(
        model.total_loss, weights)  # gradient tensors
    input_tensors = [model.inputs[0],  # input data
                     # how much to weight each sample by
                     model.sample_weights[0],
                     model.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                    ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    grads = get_gradients([X_train, np.ones(len(y_train)), y_train])
    
    metrics = {}

    for (weight, grad) in zip(weights, grads):
#         metrics["gradients/" + weight.name.split(
        metrics["gradients/" + weight.name.split(
            ':')[0] + ".gradient_mag"] = np.exp(np.mean(np.log(np.abs(grad) + 1e-9)))

    return metrics

softrelu = lambda x: relu(x, alpha=config.relu_alpha)

def make_discriminator():
    model = Sequential()
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', input_shape=(config.height, config.width, 3)))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    # 32
    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    # 16
    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    # 8
    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    model.add(Dropout(config.dis_dropout))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    input = Input(shape=(config.height, config.width, 3))
    valid = model(input)
    return Model(input, valid)

def make_generator():
    model = Sequential()
    
    kernel_size = (3, 3)
    
    model.add(Dense(128 * 16 * 16, activation='relu'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    model.add(Reshape((16, 16, 128)))
    
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    # 32
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    # 64
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization(momentum=config.batchnorm_momentum))
    model.add(Activation(softrelu))
    
    model.add(Conv2D(3, kernel_size=kernel_size, padding='same'))
    model.add(Activation('tanh'))
    
    latent = Input(shape=(config.latent_size,))
    return Model(latent, model(latent))

discriminator = make_discriminator()
generator = make_generator()

dmodel = Sequential()
dmodel.add(discriminator)
dmodel.compile(
    loss='binary_crossentropy',
    optimizer=Adam(
        lr=config.dis_learning_rate,
        beta_1=0.5
    ),
    metrics=['accuracy']
)

discriminator.trainable = False

gmodel = Sequential()
gmodel.add(generator)
gmodel.add(discriminator)
gmodel.compile(
    loss='binary_crossentropy',
    optimizer=Adam(
        lr=config.gen_learning_rate,
        beta_1=0.5
    ),
    metrics=['accuracy']
)

wandb.run.summary['graph'] = wandb.Graph.from_keras(gmodel)

real_generator = image_generator(config.batch_size, image_filenames)

example_noise = np.random.normal(0, 1, (config.examples, config.latent_size))
incorrect = np.ones((config.batch_size, 1))
correct = np.zeros((config.batch_size, 1))
 
def train_dis():
    # use noisy labels
    correct_noisy = correct + np.random.uniform(0.0, 0.1, (config.batch_size, 1))
    incorrect_noisy = incorrect - np.random.uniform(0.0, 0.1, (config.batch_size, 1))
    
    # flip a small number of labels
    flipped_idx = np.random.choice(np.arange(config.batch_size), size=int(config.flip_rate*config.batch_size))
    correct_noisy[flipped_idx] = 1 - correct_noisy[flipped_idx]
    flipped_idx = np.random.choice(np.arange(config.batch_size), size=int(config.flip_rate*config.batch_size))
    incorrect_noisy[flipped_idx] = 1 - incorrect_noisy[flipped_idx]
    
    real_imgs = next(real_generator)
    noise = np.random.normal(0, 1, (config.batch_size, config.latent_size))
    gen_imgs = generator.predict(noise)
    
    d_loss_fake, _ = dmodel.train_on_batch(gen_imgs, incorrect_noisy)
    d_loss_real, _ = dmodel.train_on_batch(real_imgs, correct_noisy)
    
    return d_loss_fake, d_loss_real, gen_imgs, real_imgs, incorrect_noisy, correct_noisy

def train_gen():
    noise = np.random.normal(0, 1, (config.batch_size, config.latent_size))
    g_loss, _ = gmodel.train_on_batch(noise, correct)
    
    for l in gmodel.layers:
        weights = l.get_weights()
        weights = [np.clip(w, -config.gen_clipvalue, config.gen_clipvalue) for w in weights]
        l.set_weights(weights)
        
    return g_loss, noise

for epoch in range(config.num_epochs):
    d_loss_fake, d_loss_real, g_loss = 0, 0, 0
    
    # train dis and gen together for N steps
    for step in tqdm(range(config.steps_per_epoch), postfix={'epoch':epoch}):
        d_loss_fake, d_loss_real, gen_imgs, real_imgs, incorrect_noisy, correct_noisy = train_dis()
        g_loss, noise = train_gen()
    
    dis_gradients_fake = calc_gradients(dmodel, gen_imgs, incorrect_noisy)
    dis_gradients_real = calc_gradients(dmodel, real_imgs, correct_noisy)
    
    gen_gradients = calc_gradients(gmodel, noise, correct)
        
    metrics = {
        'd_loss_real': d_loss_real,
        'd_loss_fake': d_loss_fake,
        'd_loss': 0.5 * np.add(d_loss_real, d_loss_fake),
        'g_loss': g_loss,
    }

    example_imgs = generator.predict(example_noise)
            
    metrics['images'] = [wandb.Image((img + 1.) * 127.5) for img in example_imgs]
    
    metrics.update({'dis_fake_' + k: v for k,v in dis_gradients_fake.items()})
    metrics.update({'dis_real_' + k: v for k,v in dis_gradients_real.items()})
    metrics.update({'gen_' + k: v for k,v in gen_gradients.items()})
            
    wandb.log(metrics)