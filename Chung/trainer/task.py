import argparse

import model  # Your model.py file.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train_data_paths',
        help = 'GCS or local path to training data',
        required = True
    )
    # Training arguments
    parser.add_argument(
        '--batch_size',
        help = 'Batch size',
        type = int,
        default = 150
    )
    parser.add_argument(
        '--hidden_units',
        help = 'Hidden layer sizes',
        nargs = '+',
        type = int,
        default = [128, 32, 4]
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    args = parser.parse_args()
    # Assign model variables to commandline arguments
    model.TRAIN_PATHS = args.train_data_paths
    model.BATCH_SIZE = args.batch_size
    model.HIDDEN_UNITS = args.hidden_units
    model.OUTPUT_DIR = args.output_dir
    # Run the training job
    model.train_and_evaluate()