import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_all = np.load('data/X.npy')
y_all = np.load('data/Y.npy')

# Maps dataset-provided label to true label
label_map = {0:9, 1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}

# Correct dataset labels
for row in range(y_all.shape[0]):
    dataset_label = np.where(y_all[row])[0][0]
    y_all[row, :] = np.zeros(10)
    y_all[row, label_map[dataset_label]] = 1

# Seed numpy rng for reproducibility
np.random.seed(1337)

# Shuffle features and targets together
rng_state = np.random.get_state()
np.random.shuffle(x_all)
np.random.set_state(rng_state)
np.random.shuffle(y_all)

# Add a dummy channel axis to input images
x_all = np.expand_dims(x_all, axis=-1)

# Center and rescale data to the range [-1, 1]
x_all = x_all - 0.5
x_all = x_all * 2

# Create a validation set from 30% of the available data
n_points = x_all.shape[0]
n_test = int(n_points * 0.3)
n_train = n_points - n_test
x_train, x_test = np.split(x_all, [n_train], axis=0)
y_train, y_test = np.split(y_all, [n_train], axis=0)

# Print important shapes in the dataset
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

### Plot the first example of each digit in the training set.
# Set up plots
plots = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
axes_list = plots[1].ravel()

for digit in range(10):
    axes = axes_list[digit]
    axes.set_axis_off()
    axes.set_title(digit)
  
    # Find the index of the first appearance of this digit
    idx = np.where(y_train[:, digit] == 1)[0][0]
    
    # Plot the image
    axes.imshow(x_train[idx, :, :, 0],
                cmap='gray')
    
# Building Pipeline
n_epochs   = 100
batch_size = 64
n_batches_per_epoch_train = n_train // batch_size
n_batches_per_epoch_test  = n_test  // batch_size
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .batch(batch_size).cache()
dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
    .batch(batch_size).cache()
train_writer = tf.summary.create_file_writer('./logs/train')
test_writer = tf.summary.create_file_writer('./logs/test')

class Dense(tf.Module):
    '''
    Creates a dense layer module.
    '''
    def __init__(self, dim_input, dim_output, do_activation=True, postfix='', name=None):
        super().__init__(name=name)
        with tf.name_scope('dense' + postfix):
            self.weights = tf.Variable(tf.initializers.he_uniform()(shape=(dim_input, dim_output), \
                                                                    dtype=tf.float32), name='weights')
            self.biases = tf.Variable(tf.zeros_initializer()(shape=(dim_output,), dtype=tf.float32), name='biases')
        self.do_activation = do_activation
        
    def __call__(self, x):
        value = tf.matmul(x, self.weights) + self.biases
        if self.do_activation:
            value = tf.nn.relu(value)
        return value
    
class Conv(tf.Module):
    '''
    Creates a convolutional layer module.
    '''
    def __init__(self, input_channels, n_filters, 
                 filter_size=3, stride=1,
                 do_activation=True, pool_size=1,
                 postfix='', name=None):
        super().__init__(name=name)
        with tf.name_scope('conv' + postfix):
            self.filters = tf.Variable(tf.initializers.he_uniform()(shape= \
                                (filter_size, filter_size, input_channels, n_filters),
                                dtype=tf.float32), name='filters')
            self.biases = tf.Variable(tf.zeros_initializer()(shape=(n_filters,), dtype=tf.float32),
                                      name='biases')
        self.n_filters = n_filters
        self.do_activation = do_activation
        self.stride = stride
        self.pool_size = pool_size
            
    def image(self, x):
        value = tf.nn.conv2d(x, self.filters, strides=(1, self.stride, self.stride, 1),
                             padding='SAME') + self.biases
        if self.do_activation:
            value = tf.nn.relu(value)
        return value
            
    def __call__(self, x):
        value = self.image(x)
        if self.pool_size > 1:
            value = tf.nn.max_pool(value, [1, self.pool_size, self.pool_size, 1],
                                   [1, self.pool_size, self.pool_size, 1],
                                   padding='VALID')
        return value
    
class ConvNet(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.conv_1 = Conv(1, 16, filter_size=5, postfix='_1')
        self.conv_2 = Conv(16, 16, filter_size=5, stride=2,
                           pool_size=2, postfix='_2')
        self.conv_3 = Conv(16, 32, pool_size=2, postfix='_3')
        self.conv_4 = Conv(32, 64, pool_size=2, postfix='_4')
        self.dense_1 = Dense(1024, 128, postfix='_1')
        self.dense_2 = Dense(128, 128, postfix='_2')
        self.dense_3 = Dense(128, 10, False, '_logits')
        
    def logits(self, x):
        x = self.conv_4(self.conv_3(self.conv_2(self.conv_1(x))))
        x = tf.reshape(x, [-1, 4 * 4 * 64])
        return self.dense_3(self.dense_2(self.dense_1(x)))
    
    def __call__(self, x):
        return tf.nn.softmax(self.logits(x), name='probabilities')
    
@tf.function
def trace_function(model, x):
    model(x)

def _loss(target, actual):
    ce_per_example = tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                             logits=actual)
    return tf.reduce_mean(ce_per_example, name='loss')

optimizer = tf.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

def train(model, x, y, i):
    with tf.GradientTape() as g:
        loss = lambda: _loss(y, model.logits(x))
    optimizer.minimize(loss, model.trainable_variables)
    predicted_digit = tf.argmax(model(x), axis=1, name='predicted_digit')
    is_equal = tf.equal(predicted_digit, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32), name='accuracy')    
    with train_writer.as_default():
        tf.summary.scalar('loss', loss(), step=i)
        tf.summary.scalar('accuracy', accuracy, step=i)
                    
def test(model, x, y, i, images=False):
    loss = _loss(y, model.logits(x))
    predicted_digit = tf.argmax(model(x), axis=1, name='predicted_digit')
    is_equal = tf.equal(predicted_digit, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32), name='accuracy')
    with test_writer.as_default():
        if images:
            value = x
            for c, conv_layer in enumerate([model.conv_1, model.conv_2, model.conv_3, model.conv_4]):
                img = conv_layer.image(value)
                value = conv_layer(value)
                for j in range(conv_layer.n_filters):
                    tf.summary.image('conv_%d/conv_%d' % (c + 1, j + 1), 
                                     tf.expand_dims(img[:, :, :, j], axis=-1),
                                     step=i)
        else:
            tf.summary.scalar('loss', loss, step=i)
            tf.summary.scalar('accuracy', accuracy, step=i)
    return accuracy

model = ConvNet()
train_batch = 0
test_batch = 0

for i in range(n_epochs):
    print('Epoch:\t', i)
    
    # Validation
    accs = []
    for x, y in dataset_test:
        accs.append(test(model, x, y, test_batch))
        test_batch += 2
    print('Val accuracy:\t', np.mean(accs))
    
    # Training run
    for x, y in dataset_train:
        if train_batch == 0:
            tf.summary.trace_on(graph=True, profiler=True)
            trace_function(model, x)
            with train_writer.as_default():
                tf.summary.trace_export(name='first_run', step=0, profiler_outdir='./logs')
        train(model, x, y, train_batch)
        # Plot images
        if train_batch % 10 == 0:
            test(model, np.expand_dims(x_test[0], axis=0), 
                 np.expand_dims(y_test[0], axis=0), test_batch, True)    
        train_batch += 1
        
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.write('./checkpoints/model')