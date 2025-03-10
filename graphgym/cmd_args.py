import argparse


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg',
                        dest='cfg_file',
                        type=str,
                        required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done',
                        action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('--save',
                        default=0,
                        help='Whether or not we save the model after training') 
    parser.add_argument('--model_path',
                        default="",
                        help='Where the model we want to retrieve comes from') 
    parser.add_argument('opts',
                    default=None,
                    nargs=argparse.REMAINDER,
                    help='See graphgym/config.py for remaining options.')
    parser.add_argument('--get_edge_weights',
                    default=0,
                    help='Do we save edge weights') 


    return parser.parse_args()
