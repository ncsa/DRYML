import os
from dryml.dry_object import DryObject, DryObjectFactory
from dryml.dry_model import DryModel

class DryRepo(object):
    def __init__(self, directory, create=False, **kwargs):
        super().__init__(**kwargs)

        # Check that directory exists
        if not os.path.exists(directory):
            if create:
                os.mkdir(directory)
        self.directory = directory
