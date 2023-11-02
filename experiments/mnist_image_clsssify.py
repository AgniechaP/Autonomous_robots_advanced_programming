from argparse import ArgumentParser
import sys
sys.path.append('/home/agnieszka/Documents/AIProject/')
import ai
import tensorflow as tf
import tqdm



def main(args):
    # todo: load mnist dataset
    train_ds, val_ds = ai.datasets.mnist(args.batch_size)

    # todo: create and optimize model (add regularization like dropout and batch normalization)
    model = ai.models.image.ImageClassifier(10)

    # todo: create optimizer (optional: try with learning rate decay)
    optimizer = tf.keras.optimizers.legacy.Adam(0.0001)

    # todo: define query function
    def query(images, labels, training):
        predictions = model(images, training)
        loss = ai.losses.classification_loss(labels, predictions)

        return loss

    # todo: define train function
    def train(images, labels):
        with tf.GradientTape() as tape:
            loss = query(images, labels, True)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    # todo: run training and evaluation for number or epochs (from argument parser)
    #  and print results (accumulated) from each epoch (train and val separately)
    img_loss = tf.metrics.Mean('classifier') 
    # Number of training epochs
    num_epochs = args.num_epochs

    for i in range(int(num_epochs)):
        img_loss.reset_states()
        with tqdm.tqdm(total=60000) as pbar:

            for images, labels in train_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5
            
                loss = train(images, labels)
                img_loss.update_state(loss)
                pbar.update(tf.shape(images)[0].numpy())
        
        print('\n============================')
        print('Train epoch')
        print(f'Classifier loss: {img_loss.result().numpy()}')
        print('============================\n')

        img_loss.reset_states()
        with tqdm.tqdm(total=10000) as pbar:
            for images, labels in val_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5

                loss = query(images, labels, False)
                img_loss.update_state(loss)
                pbar.update(tf.shape(images)[0].numpy())

        print('\n============================')
        print('Validation epoch')
        print(f'Classifier loss: {img_loss.result().numpy()}')
        print('============================\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    # todo: pass arguments
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, default=10) 
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        ai.utils.allow_memory_growth()

    main(args)
