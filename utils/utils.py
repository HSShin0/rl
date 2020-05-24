import json


class Params:
    """
    A class that loads hyperparameters from json file.
    """
    def __init__(self, json_filepath: str):
        with open(json_filepath, 'r') as f:
            self.__dict__ = json.load(f)

    @property
    def dict(self):
        return self.__dict__
