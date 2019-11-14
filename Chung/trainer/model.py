import pandas as pd
from skimage import io
import multiprocessing
from joblib import Parallel, delayed
import wget
import os
import shutil
from google.cloud import storage
import sys



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from sklearn.utils import shuffle



def model():
    
    incesnet = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3))
    x = incesnet.output
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(label_names), activation='softmax')(x) 
    model = Model(inputs=incesnet.input, outputs=predictions)

    return model

def main(job_dir,**args):

    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'

    ##Using the GPU
    with tf.device('/device:GPU:0'):

        ##Loading the data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        ##Pre processing the data
        train_labels = keras.utils.np_utils.to_categorical(train_labels, 10)
        eval_labels = keras.utils.np_utils.to_categorical(eval_labels, 10)
        train_data = np.reshape(train_data, [-1, 28, 28, 1])
        eval_data = np.reshape(eval_data, [-1,28,28,1])

        ## Initializing the model
        Model = model(train_data.shape[1:]);

        ## Compling the model
        Model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"]);

        ## Printing the modle summary
        Model.summary()

        ## Adding the callback for TensorBoard and History
        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        ##fitting the model
        Model.fit(x = train_data, y = train_labels, epochs = 4,verbose = 1, batch_size=100, callbacks=[tensorboard], validation_data=(eval_data,eval_labels) )

        # Save model.h5 on to google storage
        Model.save('model.h5')
        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)