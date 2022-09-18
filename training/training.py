import os
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
import yaml

from utils.dataset import load_dataset
from utils.experiment import ExperimentHandler, restore_from_checkpoint_latest, ds_tqdm
from utils.device import allow_memory_growth
from model.signal_transformer import SignalTransformer
from loss.triplet_loss import batch_all_triplet_loss


def main(args):
    # load data
    train_ds, val_ds = load_dataset(args.dataset_path, args.batch_size)

    model = SignalTransformer(num_signals=6,
                              num_layers=args.num_encoder_layers,
                              d_model=args.d_model,
                              num_heads=args.num_heads,
                              dff=args.dff,
                              latent_vector_size=args.lv_size,
                              input_signal_length=160)

    model.warmup()
    model.summary()

    # prepare optimization method and helpers
    eta_f = tf.keras.optimizers.schedules.ExponentialDecay(args.eta, args.eta_decay_steps, args.eta_beta)
    wd_f = tf.keras.experimental.CosineDecay(args.weight_decay, args.weight_decay_steps, args.weight_decay_alpha)

    eta = tf.Variable(args.eta)
    wd = tf.Variable(args.weight_decay)

    optimizer = AdamW(wd, eta)

    experiment_handler = ExperimentHandler(
        args.working_path, args.out_name,
        model=model,
        optimizer=optimizer
    )

    # restore if provided
    if args.restore_path is not None:
        restore_from_checkpoint_latest(
            path=args.restore_path,
            model=model,
            optimizer=optimizer
        )

    def query(signals, positions, training):
        latent_vectors = model(signals, tf.convert_to_tensor(training))

        triplet_loss, fraction_tl = batch_all_triplet_loss(positions, latent_vectors,
                                                           margin=args.margin,
                                                           dist_threshold=args.dist_threshold)
        return triplet_loss, latent_vectors, fraction_tl

    def train_step_fn(signals, positions):
        with tf.GradientTape() as tape:
            loss, latent_vectors, fraction_tl = query(signals, positions, True)
            total_loss = tf.reduce_mean(loss)

        t_vars = model.trainable_variables
        grads = tape.gradient(total_loss, t_vars)
        optimizer.apply_gradients(zip(grads, t_vars))

        return loss, latent_vectors, fraction_tl

    # run training and validation
    epoch = 0
    train_step, val_step = 0, 0
    best_result = None
    late_stop = 0
    mean_loss = tf.metrics.Mean('triplet_loss')
    mean_fraction_valid_triplets = tf.metrics.Mean('fraction_valid_triplets')

    while True:
        experiment_handler.log_training()

        mean_loss.reset_states()
        mean_fraction_valid_triplets.reset_states()
        for i, signals, positions in ds_tqdm('Train', train_ds, epoch, args.batch_size):

            eta.assign(eta_f(train_step))
            wd.assign(wd_f(train_step))

            loss, latent_vectors, fraction_tl = train_step_fn(signals, positions)
            mean_loss(loss)
            mean_fraction_valid_triplets(fraction_tl)

            if train_step % args.log_interval == 0:
                tf.summary.scalar('info/eta', eta, step=train_step)
                tf.summary.scalar('info/weight_decay', wd, step=train_step)
                tf.summary.scalar('metrics/triplet_loss', tf.reduce_mean(loss), step=train_step)
                tf.summary.scalar('metrics/fraction_valid_triplets', fraction_tl, step=train_step)

            train_step += 1

        result_loss = mean_loss.result()
        result_fraction_valid_triplets = mean_fraction_valid_triplets.result()
        tf.summary.scalar('epoch/triplet_loss', result_loss, step=epoch)
        tf.summary.scalar('epoch/fraction_valid_triplets', result_fraction_valid_triplets, step=epoch)

        experiment_handler.save_last()
        experiment_handler.flush()
        experiment_handler.log_validation()

        mean_loss.reset_states()
        mean_fraction_valid_triplets.reset_states()
        for i, signals, positions in ds_tqdm('Validation', val_ds, epoch, args.batch_size):
            loss, latent_vectors, fraction_tl = query(signals, positions, False)
            mean_loss(loss)
            mean_fraction_valid_triplets(fraction_tl)

            if val_step % args.log_interval == 0:
                tf.summary.scalar('metrics/triplet_loss', tf.reduce_mean(loss), step=val_step)
                tf.summary.scalar('metrics/fraction_valid_triplets', fraction_tl, step=val_step)

            val_step += 1

        result_loss = mean_loss.result()
        result_fraction_valid_triplets = mean_fraction_valid_triplets.result()
        tf.summary.scalar('epoch/triplet_loss', result_loss, step=epoch)
        tf.summary.scalar('epoch/fraction_valid_triplets', result_fraction_valid_triplets, step=epoch)

        experiment_handler.flush()

        if best_result is None or result_loss < best_result:
            experiment_handler.save_best()
            model.save_weights(os.path.join(experiment_handler.export_model_path, 'model'))
            best_result = result_loss
            late_stop = 0
        elif epoch >= args.num_epochs > 0:
            late_stop += 1
            if late_stop > args.late_stop_threshold:
                break
        else:
            late_stop = 0

        epoch += 1


if __name__ == '__main__':
    import os

    path = os.getcwd()

    print(path)
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--working-path', type=str, default='./workspace')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--eta-decay-steps', type=int, default=100)
    parser.add_argument('--eta-beta', type=float, default=0.99)
    parser.add_argument('--num-epochs', type=int, default=-1)
    parser.add_argument('--late-stop-threshold', type=int, default=-1)
    parser.add_argument('--out-name', type=str, required=True)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--weight-decay-steps', type=float, default=20000)
    parser.add_argument('--weight-decay-alpha', type=float, default=1e-3)
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    parser.add_argument('--dist-threshold', type=float, default=0.25)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--num-encoder-layers', type=int, default=8)
    parser.add_argument('--d-model', type=int, default=16)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dff', type=int, default=2048)
    parser.add_argument('--lv-size', type=int, default=256)
    args, _ = parser.parse_known_args()

    config = {
        'dist_threshold': args.dist_threshold,
        'margin': args.margin,
        'num_encoder_layers': args.num_encoder_layers,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'dff': args.dff,
        'lv_size': args.lv_size
    }

    os.mkdir(os.path.join(args.working_path, args.out_name))
    with open(os.path.join(args.working_path, args.out_name, 'model_config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    if args.allow_memory_growth:
        allow_memory_growth()

    main(args)
