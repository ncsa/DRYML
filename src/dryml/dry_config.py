import os
import json
from typing import Union, IO

class DryConfig(dict):
    def save(file: Union[str, IO[bytes]]) -> bool:
        if type(file) is str:
            with open(file, 'w') as f:
                f.write(json.dumps(self))
        else:
            file.write(json.dumps(self))
        return True

    def get_hash_str(self):
        return json.dumps(self)

    def get_hash(self):
        return hash(self.get_hash_str())
