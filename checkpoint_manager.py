import os
import json
import time
import tensorflow as tf

DEFAULT_EXPORT_DIR = "./"


class CheckpointManager:
    def __init__(self, number):
        self.number = number
        self.checkpoint_path = DEFAULT_EXPORT_DIR + str(number) + '/'
        self.model_path = self.checkpoint_path + 'model/'
        self.status = 'READY'

    def on_training_start(self, dataset, n_epochs, batch_size, learning_rate, optimizer, dropout):
        self.status = 'RUNNING'
        self.started_at = time.asctime()

        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.dropout = dropout

        self.started_at = time.asctime()
        self.finished_at = None
        self.score = None

        self.training_batch_losses = []
        self.training_batch_accuracies = []

        self._write_current_status()

    def on_epoch_completed(self):
        pass

    def on_batch_completed(self, training_batch_loss, training_batch_accuracy):
        self.training_batch_losses.append(float(training_batch_loss))
        self.training_batch_accuracies.append(float(training_batch_accuracy))
        self._write_current_status()

    def on_error(self):
        self.status = 'FAILED'
        self.finished_at = time.asctime()
        self._write_current_status()


    def on_training_completed(self, score):
        self.status = 'FINISHED'
        self.finished_at = time.asctime()
        self.score = "{:3.1%}".format(score)
        self._write_current_status()

    def save_model(self, session, step=0):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        saver = tf.train.Saver()
        saver.save(session, self.model_path + 'model.ckpt', step)

    def restore_model(self, session):
        saver = tf.train.Saver()
        meta = tf.train.import_meta_graph(self.model_path + "model.ckpt-0.meta")
        meta.restore(session, tf.train.latest_checkpoint(self.model_path))

    def _write_current_status(self):
        if not os.path.exists(DEFAULT_EXPORT_DIR):
            os.makedirs(DEFAULT_EXPORT_DIR)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        json_string = self.generate_json()

        with open(self.checkpoint_path + "checkpoint_{}.json".format(self.number), "w") as out_file:
            out_file.write(json_string)


    def generate_json(self):
        return json.dumps({
            'number': self.number,
            'started_at': self.started_at,
            'status': self.status,
            'finished_at': self.finished_at,
            'score': self.score,
            'dataset': self.dataset,
            'hyper_parameters': {
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer,
                'dropout': self.dropout
            },
            'training_batch_losses': self.training_batch_losses,
            'training_batch_accuracies': self.training_batch_accuracies,
        }, indent=4)

def main():
    pass

if __name__ == '__main__':
    main()
