from dryml.dry_object import DryObject, DryObjectFile, DryObjectDef
from dryml.utils import is_nonstring_iterable, is_dictlike, get_class_str
from typing import Union, Callable, Type, Mapping


class DrySelector(object):
    "Utility object for selecting for specific dry objects"

    @staticmethod
    def from_def(
            obj_def: DryObjectDef,
            verbosity: int = 0,
            cls_str_compare: bool = True):
        return DrySelector(
            obj_def['cls'],
            args=obj_def['dry_args'],
            kwargs=obj_def['dry_kwargs'],
            verbosity=verbosity,
            cls_str_compare=cls_str_compare)

    @staticmethod
    def from_dict(
            obj_dict: Mapping,
            verbosity: int = 0,
            cls_str_compare: bool = True):
        args = obj_dict.get('dry_args', ())
        kwargs = obj_dict.get('dry_kwargs', {})
        return DrySelector(
            obj_dict['cls'],
            args=args,
            kwargs=kwargs,
            verbosity=verbosity,
            cls_str_compare=cls_str_compare)

    @staticmethod
    def from_obj(
            obj: DryObject,
            verbosity: int = 0,
            cls_str_compare: bool = True):
        return DrySelector.from_def(
            obj.definition(),
            verbosity=verbosity,
            cls_str_compare=cls_str_compare)

    @staticmethod
    def build(
            obj,
            verbosity: int = 0,
            cls_str_compare: bool = True):
        if isinstance(obj, DryObject):
            return DrySelector.from_obj(
                obj,
                verbosity=verbosity,
                cls_str_compare=cls_str_compare)
        elif isinstance(obj, DryObjectDef):
            return DrySelector.from_def(
                obj,
                verbosity=verbosity,
                cls_str_compare=cls_str_compare)
        elif isinstance(obj, Mapping):
            return DrySelector.from_dict(
                obj,
                verbosity=verbosity,
                cls_str_compare=cls_str_compare)
        else:
            raise TypeError(
                f"Can't construct DrySelector from type {type(obj)}")

    def __init__(self, cls: Type, args=None, kwargs=None,
                 verbosity: int = 0, cls_str_compare: bool = True):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.verbosity = verbosity

    @staticmethod
    def match_objects(key_object, value_object):
        if issubclass(type(key_object), type):
            # We have a type object. They match if
            return issubclass(key_object, value_object)
        elif callable(key_object):
            return key_object(value_object)
        elif is_dictlike(key_object):
            # dictlike branch is first because dictlike objects
            # are also iterable objects
            if not is_dictlike(value_object):
                return False
            for key in key_object:
                if key not in value_object:
                    return False
                if not DrySelector.match_objects(key_object[key],
                                                 value_object[key]):
                    return False
        elif is_nonstring_iterable(key_object):
            if not is_nonstring_iterable(value_object):
                return False
            # For now, lists must have the same number of elements
            if len(key_object) != len(value_object):
                return False
            for i in range(len(key_object)):
                # Each object must match
                if not DrySelector.match_objects(key_object[i],
                                                 value_object[i]):
                    return False
        else:
            if key_object != value_object:
                return False

        return True

    def cls_compare(self, matcher, cls):
        matched = True
        if isinstance(matcher, type):
            if not issubclass(cls, matcher):
                matched = False
        elif isinstance(matcher, Callable):
            if not matcher(cls):
                matched = False
        elif isinstance(matcher, str):
            if matcher != get_class_str(cls):
                matched = False
        else:
            raise ValueError(f"Unsupported matcher type: {type(matcher)}")
        if not matched:
            # Failure message
            if self.verbosity > 0:
                print("Class doesn't match")
            if self.verbosity > 1:
                print(f"Got {type(cls)}")
            return False
        else:
            return True

    def args_compare(self, matcher, args):
        if DrySelector.match_objects(matcher, args):
            return True
        else:
            if self.verbosity > 0:
                print("Args don't match")
            if self.verbosity > 1:
                print(f"Got {args}")
            return False

    def kwargs_compare(self, matcher, kwargs):
        if DrySelector.match_objects(matcher, kwargs):
            return True
        else:
            if self.verbosity > 0:
                print("Kwargs don't match")
            if self.verbosity > 1:
                print(f"Got {kwargs}")
            return False

    def __call__(
            self,
            obj: Union[DryObject, DryObjectFile, DryObjectDef, Mapping]):
        # Get definition
        if isinstance(obj, DryObjectDef):
            obj_def = obj
        elif isinstance(obj, DryObject) or isinstance(obj, DryObjectFile):
            obj_def = obj.definition()
        else:
            obj_def = DryObjectDef.from_dict(obj)

        # If required, check object class
        if self.cls is not None:
            if not self.cls_compare(self.cls, obj_def.cls):
                if self.verbosity > 0:
                    print("Class didn't match")
                if self.verbosity > 1:
                    print(f"Expected class {self.cls} got {obj_def.cls}")
                return False

        # Check object args
        if self.args is not None:
            if not self.args_compare(self.args, obj_def.args):
                if self.verbosity > 0:
                    print("Args didn't match")
                if self.verbosity > 1:
                    print(f"Expected args {self.args} got {obj_def.args}")
                return False

        # Check object kwargs
        if self.kwargs is not None:
            if not self.kwargs_compare(self.kwargs, obj_def.kwargs):
                if self.verbosity > 0:
                    print("Kwargs didn't match")
                if self.verbosity > 1:
                    print(f"Expected kwargs {self.kwargs} "
                          f"got {obj_def.kwargs}")
                return False

        return True
