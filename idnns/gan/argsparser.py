from argparse import ArgumentParser


class ArgsParser:

    def __init__(self):
        self.parser = ArgumentParser(description='Process some integers.')
        self.parser.add_argument('-batch_size', '-b', dest="batch_size", default=100, type=int, help='The size of the batch')
        self.parser.add_argument('-train_epoch', '-e', dest="train_epoch", default=20, type=int, help='The size of the batch')
        self.parser.add_argument('-learning_rate', '-lr', dest="learning_rate", default=0.0002, type=float, help='The size of the batch')

    def parse(self):
        return self.parser.parse_args()