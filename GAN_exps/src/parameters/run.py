from argparse import ArgumentParser


class RuntimeParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('--cuda', type=int, default=3)