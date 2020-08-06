import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run Model.")

    # Data Parameters
    parser.add_argument("--train-path",
                        nargs="?",
                        default="../data/train",
                        help="Training path.")

    parser.add_argument("--validation-path",
                        nargs="?",
                        default="../data/validation",
                        help="Validation path.")

    parser.add_argument("--test-path",
                        nargs="?",
                        default="../data/test",
                        help="Testing path.")

    parser.add_argument("--TR-option",
                        nargs="?",
                        default="T",
                        help="Training or Restore pattern. (T or R).")

    # Stock Hyperparameters
    parser.add_argument("--stock-names",
                        type=list,
                        default=['Aluminium', 'Copper', 'Lead', 'Nickel', 'Tin', 'Zinc'],
                        help="Stock name.")

    parser.add_argument("--max-n-days",
                        type=int,
                        default=20,
                        help="Max n days.")

    # Model Hyperparameters
    parser.add_argument("--stock",
                        nargs="?",
                        default='Aluminium',
                        help="Stock name.")

    parser.add_argument("--features-dim",
                        type=int,
                        default=25,
                        help="Dimensionality of features.")

    parser.add_argument("--embedding-dim",
                        type=int,
                        default=32,
                        help="Dimensionality of character embedding. (default: 100)")

    parser.add_argument("--filter-sizes",
                        type=list,
                        default=[3, 5, 7],
                        help="Filter sizes.")

    parser.add_argument("--conv-padding-sizes",
                        type=list,
                        default=[1, 2, 3],
                        help="Padding sizes for Conv Layer.")

    parser.add_argument("--dilation-sizes",
                        type=list,
                        default=[1, 2, 3],
                        help="Dilation sizes for Conv Layer.")

    parser.add_argument("--num-filters",
                        type=list,
                        default=[32, 32, 32],
                        help="Number of filters per filter size. (default: 128)")

    parser.add_argument("--pooling-size",
                        type=int,
                        default=3,
                        help="Pooling sizes. (default: 3)")

    parser.add_argument("--rnn-dim",
                        type=int,
                        default=128,
                        help="Dimensionality for RNN Neurons. (default: 256)")

    parser.add_argument("--rnn-type",
                        nargs="?",
                        default="GRU",
                        help="Type of RNN Cell. ('RNN', 'LSTM', 'GRU')")

    parser.add_argument("--rnn-layers",
                        type=int,
                        default=1,
                        help="Number of RNN Layers. (default: 1)")

    parser.add_argument("--skip-size",
                        type=int,
                        default=3,
                        help="Skip window of Skip-RNN Layers. (default: 3)")

    parser.add_argument("--skip-dim",
                        type=int,
                        default=5,
                        help="Dimensionality for Skip-RNN Layers. (default: 5)")

    parser.add_argument("--fc-dim",
                        type=int,
                        default=16,
                        help="Dimensionality for FC Neurons. (default: 512)")

    parser.add_argument("--dropout-rate",
                        type=float,
                        default=0.5,
                        help="Dropout keep probability. (default: 0.5)")

    # Training Parameters
    parser.add_argument("--epochs",
                        type=int,
                        default=30,
                        help="Number of training epochs. (default: 30)")

    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Batch Size. (default: 32)")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="Learning rate. (default: 0.001)")

    parser.add_argument("--decay-rate",
                        type=float,
                        default=0.95,
                        help="Rate of decay for learning rate. (default: 0.95)")

    parser.add_argument("--decay-steps",
                        type=int,
                        default=500,
                        help="How many steps before decay learning rate. (default: 500)")

    parser.add_argument("--norm-ratio",
                        type=float,
                        default=1.25,
                        help="The ratio of the sum of gradients norms of trainable variable. (default: 1.25)")

    parser.add_argument("--l2-lambda",
                        type=float,
                        default=0.0,
                        help="L2 regularization lambda. (default: 0.0)")

    parser.add_argument("--num-checkpoints",
                        type=int,
                        default=3,
                        help="Number of checkpoints to store. (default: 5)")

    return parser.parse_args()