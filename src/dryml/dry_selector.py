from dryml.dry_object import DryObject

class DrySelector(object):
    "Utility object for selecting for specific dry objects"
    def __init__(self, cls=None, args=None, kwargs=None):
         self.cls = cls
         self.args = args
         self.kwargs = kwargs

    @staticmethod
    def match_objects(key_object, value_object):
        if callable(key_object):
            print("Callable branch")
            return key_object(value_object)
        if issubclass(type(key_object), list):
            print("list branch")
            if not issubclass(type(value_object), list):
                return False
            # For now, lists must have the same number of elements
            if len(key_object) != len(value_object):
                return False
            for i in range(len(key_object)):
                if not DrySelector.match_objects(key_object[i], value_object[i]):
                    return False
        if issubclass(type(key_object), dict):
            print("dict branch")
            if not issubclass(type(value_object), dict):
                print(f"value type: {type(value_object)}")
                return False
            for key in key_object:
                print(f"checking key {key}")
                if key not in value_object:
                    print(f"key doesn't exist")
                    return False
                if not DrySelector.match_objects(key_object[key], value_object[key]):
                    print(f"values didn't match")
                    return False
        print("Value path")
        if key_object != value_object:
            return False
        else:
            return True

    def __call__(self, obj: DryObject):
        # If required, check object class
        if self.cls is not None:
            print("Checking class")
            if not DrySelector.match_objects(self.cls, type(obj)):
                return False
            print("class matches")

        # Check object args
        if self.args is not None:
            print("checking args")
            if not DrySelector.match_objects(self.args, obj.dry_args):
                return False
            print("args match")

        # Check object kwargs
        if self.kwargs is not None:
            print("checking kwargs")
            if not DrySelector.match_objects(self.kwargs, obj.dry_kwargs):
                return False
            print("kwargs match")

        return True
