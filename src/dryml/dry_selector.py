from dryml.dry_object import DryObject
from dryml.utils import is_nonstring_iterable, is_dictlike

class DrySelector(object):
    "Utility object for selecting for specific dry objects"
    def __init__(self, cls=None, args=None, kwargs=None):
         self.cls = cls
         self.args = args
         self.kwargs = kwargs

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

    def __call__(self, obj: DryObject):
        # If required, check object class
        if self.cls is not None:
            if not isinstance(obj, self.cls):
                return False

        # Check object args
        if self.args is not None:
            if not DrySelector.match_objects(self.args, obj.dry_args):
                return False

        # Check object kwargs
        if self.kwargs is not None:
            if not DrySelector.match_objects(self.kwargs, obj.dry_kwargs):
                return False

        return True
