from dryml.dry_object import DryObject, DryObjectFile
from dryml.utils import is_nonstring_iterable, is_dictlike
from typing import Union

class DrySelector(object):
    "Utility object for selecting for specific dry objects"
    def __init__(self, cls=None, args=None, kwargs=None, verbosity:int=0):
         self.cls = cls
         self.args = args
         self.kwargs = kwargs
         self.verbosity = verbosity

    @staticmethod
    def match_objects(key_object, value_object):
        if callable(key_object):
            return key_object(value_object)
        elif is_dictlike(key_object):
            # dictlike branch is first because dictlike objects are also iterable objects
            if not is_dictlike(value_object):
                return False
            for key in key_object:
                if key not in value_object:
                    return False
                if not DrySelector.match_objects(key_object[key], value_object[key]):
                    return False
        elif is_nonstring_iterable(key_object):
            if not is_nonstring_iterable(value_object):
                return False
            # For now, lists must have the same number of elements
            if len(key_object) != len(value_object):
                return False
            for i in range(len(key_object)):
                if not DrySelector.match_objects(key_object[i], value_object[i]):
                    return False
        else:
            if key_object != value_object:
                return False

        return True

    def __call__(self, obj: Union[DryObject, DryObjectFile]):
        if isinstance(obj, DryObject):
            if self.verbosity > 1:
                print(f"selector Dry object branch")
            # If required, check object class
            if self.cls is not None:
                if not isinstance(obj, self.cls):
                    if self.verbosity > 0:
                        print("Class doesn't match")
                    if self.verbosity > 1:
                        print(f"Got {type(obj)}")
                    return False

            # Check object args
            if self.args is not None:
                if not DrySelector.match_objects(self.args, obj.dry_args):
                    if self.verbosity > 0:
                        print("Args don't match")
                    if self.verbosity > 1:
                        print(f"Got {obj.dry_args}")
                    return False

            # Check object kwargs
            if self.kwargs is not None:
                if not DrySelector.match_objects(self.kwargs, obj.dry_kwargs):
                    if self.verbosity > 0:
                        print("Kwargs don't match")
                    if self.verbosity > 1:
                        print(f"Got {obj.dry_kwargs}")
                    return False

        elif isinstance(obj, DryObjectFile):
            if self.verbosity > 1:
                print(f"selector other type branch")
            # If required, check object class
            if self.cls is not None:
                if self.cls != obj.cls:
                    if self.verbosity > 0:
                        print("Class doesn't match")
                    if self.verbosity > 1:
                        print(f"Got {obj.cls}")
                    return False

            # Check object args
            if self.args is not None:
                if not DrySelector.match_objects(self.args, obj.args):
                    if self.verbosity > 0:
                        print("Args don't match")
                    if self.verbosity > 1:
                        print(f"Got {obj.args}")
                    return False

            # Check object kwargs
            if self.kwargs is not None:
                if not DrySelector.match_objects(self.kwargs, obj.kwargs):
                    if self.verbosity > 0:
                        print("Kwargs don't match")
                    if self.verbosity > 1:
                        print(f"Got {obj.kwargs}")
                    return False

        return True
