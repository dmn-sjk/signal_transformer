import io
import os
import zipfile

import requests
import tensorflow as tf
from tqdm import tqdm


class ExperimentHandler(object):

    def __init__(self, working_path, out_name, max_to_keep=3, **objects_to_save):
        super(ExperimentHandler, self).__init__()

        # prepare log writers
        train_log_path = _get_or_create_dir(working_path, out_name, 'logs', 'train')
        val_log_path = _get_or_create_dir(working_path, out_name, 'logs', 'val')
        self.export_model_path = _get_or_create_dir(working_path, out_name, 'export_model')
        self.extras_path = _get_or_create_dir(working_path, out_name, 'extras')

        self.train_writer = tf.summary.create_file_writer(train_log_path)
        self.val_writer = tf.summary.create_file_writer(val_log_path)

        # prepare checkpoints
        self.last_path = _get_or_create_dir(working_path, out_name, 'checkpoints', 'last')
        self.best_path = _get_or_create_dir(working_path, out_name, 'checkpoints', 'best')

        self.checkpoint_last, self.checkpoint_manager_last = _prepare_checkpoint_manager(
            self.last_path, max_to_keep,
            **objects_to_save
        )

        self.checkpoint_best, self.checkpoint_manager_best = _prepare_checkpoint_manager(
            self.best_path, max_to_keep,
            **objects_to_save
        )

    def log_training(self):
        self.train_writer.set_as_default()

    def log_validation(self):
        self.val_writer.set_as_default()

    def flush(self):
        self.train_writer.flush()
        self.val_writer.flush()

    def save_last(self):
        self.checkpoint_manager_last.save()

    def save_best(self):
        self.checkpoint_manager_best.save()

    def restore_best(self):
        self.checkpoint_best.restore(self.checkpoint_manager_best.latest_checkpoint)

    def restore(self, path):
        self.checkpoint_last.restore(tf.train.latest_checkpoint(path)).assert_consumed()


def restore_from_checkpoint(path, **kwargs):
    checkpoint = tf.train.Checkpoint(**kwargs)
    return checkpoint.restore(path)


def restore_from_checkpoint_latest(path, **kwargs):
    return restore_from_checkpoint(tf.train.latest_checkpoint(path), **kwargs)


def _prepare_checkpoint_manager(path, max_to_keep, **kwargs):
    checkpoint = tf.train.Checkpoint(**kwargs)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=path,
        max_to_keep=max_to_keep
    )
    return checkpoint, checkpoint_manager


def _get_or_create_dir(*paths):
    join_path = os.path.join(*paths)
    if not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path


def _tqdm_template(t, i, s=None):
    if s is None:
        bar = '%s epoch %d | Elapsed: {elapsed} | Rate: {rate_fmt} | Inverted Rate: {rate_inv_fmt}' % (t, i)
    else:
        bar = '%s epoch %d | {l_bar}{bar} %s' % (t, i, '| Remaining: {remaining} | Inverted Rate: {rate_inv_fmt}')

    return tqdm(ncols=80, total=s, bar_format=bar)


def ds_tqdm(title, ds, i, batch_size, ds_size=None):
    with _tqdm_template(title, i, ds_size) as pbar:
        for i, data in enumerate(ds):
            if not isinstance(data, tuple):
                data = (data,)
            yield (i,) + data
            pbar.update(batch_size)
