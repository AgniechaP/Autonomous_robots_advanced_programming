from argparse import ArgumentParser
import sys
sys.path.append('/home/agnieszka/Documents/AIProject/')
import ai
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt 
import numpy as np 

tf.keras.backend.clear_session()

def display_images_with_predictions(dataset, model):
    num_images_to_display = 10  # Adjust the number of images to display

    for images, labels in dataset.take(1):
        predicted_labels = model(images, training=False).numpy()
        predicted_classes = np.argmax(predicted_labels, axis=1)

        plt.figure(figsize=(12, 8))
        for i in range(num_images_to_display):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')
            plt.title(f'Predicted: {predicted_classes[i]}, Actual: {labels[i].numpy()}')
            plt.axis('off')

        plt.show()



def main(args):
    # todo: load mnist dataset
    train_ds, val_ds = ai.datasets.mnist(args.batch_size)

    # todo: create and optimize model (add regularization like dropout and batch normalization)
    model = ai.models.image.FlatImageClassifier(10)

    # To avoid error of logits.shape tf.keras.layers.Flatten() was added to classifier.py 
    # Optionally BatchNormalization() was also added to classifier.py (FlatImageClassifier)

    # todo: create optimizer (optional: try with learning rate decay)
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9, staircase=False, name=None)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_decay)

     # Number of training epochs
    num_epochs = args.num_epochs

    # todo: define query function
    # @tf.function
    def query(images, labels, training):
        predictions = model(images, training)
        loss = ai.losses.classification_loss(labels, predictions)

        return loss
    
    # todo: define train function
    # @tf.function
    def train(images, labels):
        with tf.GradientTape() as tape:
            loss = query(images, labels, True)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss
    
    # todo: run training and evaluation for number or epochs (from argument parser)
    #  and print results (accumulated) from each epoch (train and val separately)
    mc_loss = tf.metrics.Mean('classifier')
    
    for i in range(int(num_epochs)):
        mc_loss.reset_states()
        with tqdm.tqdm(total=60000) as pbar:

            for images, labels in train_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5
            
                loss = train(images, labels)
                mc_loss.update_state(loss)
                pbar.update(tf.shape(images)[0].numpy())

        print('\n============================')
        print('Train epoch')
        print(f'Classifier loss: {mc_loss.result().numpy()}')
        print('============================\n')

        display_images_with_predictions(train_ds, model)

        mc_loss.reset_states()
        with tqdm.tqdm(total=10000) as pbar:
            for images, labels in val_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5

                loss = query(images, labels, False)
                mc_loss.update_state(loss)
                pbar.update(tf.shape(images)[0].numpy())

        print('\n============================')
        print('Validation epoch')
        print(f'Classifier loss: {mc_loss.result().numpy()}')
        print('============================\n')

        display_images_with_predictions(val_ds, model)


if __name__ == '__main__':
    parser = ArgumentParser()
    # todo: pass arguments
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    # Added arguments
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, default=10) 
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        ai.utils.allow_memory_growth()

    main(args)

# To run the script:
# python3 experiments/mnist_flat_clsssify.py --batch-size 10
