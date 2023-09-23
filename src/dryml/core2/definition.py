from copy import deepcopy
from collections.abc import ItemsView
from boltons.iterutils import remap, is_collection, PathAccessError, default_enter, default_exit
import hashlib
from inspect import isclass
import numpy as np
from pprint import pprint

from dryml.core2.util import is_dictlike, cls_super, \
    get_class_str, is_nonclass_callable, hashval_to_digest, \
    digest_to_hashval, get_remember_view
from dryml.core2.repo import manage_repo


def definition_enter(path, key, value):
    if isinstance(value, Definition):
        return {}, ItemsView(value)
    else:
        return default_enter(path, key, value)


def definition_exit(path, key, value, new_parent, new_items):
    if isinstance(value, Definition):
        # We have a definition. we have to be careful how we construct it.
        for k, v in new_items:
            new_parent[k] = v
        args = new_parent['args']
        kwargs = new_parent['kwargs']
        cls = new_parent['cls']
        return type(value)(cls, *args, **kwargs)
    else:
        return default_exit(path, key, value, new_parent, new_items)


def deepcopy_skip_definition_object(defn):
    def _deepcopy_enter(path, key, value):
        from dryml.core2.object import Object
        if isinstance(value, Object):
            return value, False
        elif isinstance(value, Definition):
            return value, False
        else:
            return default_enter(path, key, value)

    def _deepcopy_visit(path, key, value):
        from dryml.core2.object import Object
        if isinstance(value, Object):
            # We have an already realized class instance. We shouldn't deep copy it.
            return key, value
        elif isinstance(value, Definition):
            # unique definitions are supposed to refer to specific objects during 'rendering'
            # We shouldn't copy Definitions
            return key, value
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
        else:
            return key, deepcopy(value)

    if type(defn) is Definition:
        return remap([defn], enter=_deepcopy_enter, visit=_deepcopy_visit, exit=definition_exit)[0]
    else:
        return remap(defn, enter=_deepcopy_enter, visit=_deepcopy_visit, exit=definition_exit)


# Special value to skip args
SKIP_ARGS = object()


class Definition(dict):
    allowed_keys = ['cls', 'args', 'kwargs']
    def __init__(self, *args, **kwargs):
        init = False
        if len(args) > 0:
            if not callable(args[0]) and not isclass(args[0]):
                raise ValueError("First positional argument must be a class or callable.")
            if len(args) > 1 and args[1] is SKIP_ARGS:
                if len(args) > 2:
                    raise ValueError("SKIP_ARGS must be the only positional argument besides the class.")
                super().__init__(
                    cls=args[0],
                    kwargs=kwargs)
                init = True
            else:
                super().__init__(
                    cls=args[0],
                    args=args[1:],
                    kwargs=kwargs)
                init = True

        if not init:
            super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key not in self.allowed_keys:
            raise KeyError(f"Key {key} not allowed in Definition. Allowed keys are {self.allowed_keys}")
        super().__setitem__(key, value)

    def copy(self):
        # A true deepcopy
        return deepcopy(self)

    def build(self, repo=None, path=None):
        return build_from_definition(self, repo=repo, path=path)

    def __call__(self, other_def, **kwargs):
        from dryml.core2.object import Remember
        if not isinstance(other_def, Definition) and \
                not isinstance(other_def, Remember):
            raise TypeError("Definition can only be called on other Definition objects and Remember objects")
        return selector_match(self, other_def, **kwargs)

    @property
    def skip_args(self):
        if 'args' not in self:
            return True
        else:
            return False

    def __eq__(self, rhs):
        if type(self) != type(rhs):
            return False
        # We actually need to check in both directions.
        if not selector_match(self, rhs, strict=True):
            return False
        if not selector_match(rhs, self, strict=True):
            return False
        return True

    def __ne__(self, rhs):
        return not self.__eq__(rhs)

    def concretize(self):
        return concretize_definition(self)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    @property
    def cls(self):
        return self['cls']

    @property
    def args(self):
        return self['args']

    @property
    def kwargs(self):
        return self['kwargs']


def concretize_definition(defn: Definition):
    # Cache for completed results. All duplicate ConcreteDefinitions should refer to the SAME object and so the same definition
    # Key for this cache will be the ConcreteDefinition hash.
    definition_cache = {}

    def _concretize_definition_enter(path, key, value):
        if id(value) in definition_cache:
            return value, False
        elif type(value) is ConcreteDefinition:
            # The definition is already concrete. don't enter it.
            return value, False
        else:
            return definition_enter(path, key, value)

    def _concretize_definition_visit(path, key, value):
        from dryml.core2.object import Object
        if id(value) in definition_cache: 
            return key, definition_cache[id(value)]
        elif type(value) is ConcreteDefinition:
            # Value is already Concrete
            return key, value
        elif isinstance(value, Object):
            # We have an already realized class instance. We shouldn't deep copy it.
            return key, value
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
        else:
            return key, deepcopy(value)

    def _concretize_definition_exit(path, key, values, new_parent, new_items):
        if isinstance(values, Definition):
            for k, v in new_items:
                new_parent[k] = v
            try:
                args = new_parent['args']
            except KeyError:
                raise ValueError("Definition {values} which skiped arguments isn't concretizable.")
            kwargs = new_parent['kwargs']
            cls = new_parent['cls']
            # Do argument manipulations
            args, kwargs = cls_super(cls).__arg_manipulation__(*args, **kwargs)
            # Copy args so modifications to this ConcreteDefinition doesn't change the original
            # Values in the original Definitions
            args = deepcopy_skip_definition_object(args)
            kwargs = deepcopy_skip_definition_object(kwargs)
            # Create the now concrete definition
            new_def = ConcreteDefinition(cls, *args, **kwargs) 
            # Check if we've encountered this definition before
            definition_cache[id(values)] = new_def
            return new_def
        else:
            return default_exit(path, key, values, new_parent, new_items)

    if isinstance(defn, Definition):
        return remap(
            [defn],
            enter=_concretize_definition_enter,
            visit=_concretize_definition_visit,
            exit=_concretize_definition_exit)[0]
    else:
        return remap(
            defn,
            enter=_concretize_definition_enter,
            visit=_concretize_definition_visit,
            exit=_concretize_definition_exit)


class ConcreteDefinition(Definition):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError("ConcreteDefinition must be created with arguments")
        if len(args) > 0:
            if not isclass(args[0]):
                raise TypeError("ConcreteDefinition's first argument must be a class")
        super().__init__(*args, **kwargs)

    def concretize(self):
        return self

    def __hash__(self):
        return digest_to_hashval(hash_function(self))


def hash_function(structure):
    # Definition hash support
    class HashHelper(object):
        def __init__(self, the_hash):
            self.hash = the_hash

    def hash_value(value):
        # Hashes for supported values here
        try:
            hash_val = hash(value)
            return hashval_to_digest(hash_val)
        except TypeError:
            pass

        if isclass(value):
            return str(value.__qualname__)
        elif isinstance(value, np.ndarray):
            # Hash for numpy arrays
            return hashlib.sha256(value.tobytes()).hexdigest()
        else:
            raise TypeError(f"Value of type {type(value)} not supported for hashing.")

    def hash_visit(path, key, value):
        # Skip if it's a hashlib hasher
        if isinstance(value, HashHelper):
            return key, value.hash
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
    
        return key, hash_value(value)

    def hash_exit(path, key, old_parent, new_parent, new_items):
        # At this point, all items should be hashes

        # sort the items. format is [(key, value)]
        new_items = sorted(new_items, key=lambda x: x[0])

        # Combine the hashes
        hasher = hashlib.sha256()
        # Add a string representation of the old parent type
        hasher.update(type(old_parent).__qualname__.encode())
        for _, v in new_items:
            hasher.update(v.encode())
        new_hash = hasher.hexdigest()

        return HashHelper(new_hash)

    return remap(structure, enter=definition_enter, visit=hash_visit, exit=hash_exit).hash


# Creating definitions from objects
def build_definition(obj):
    instance_cache = {}
    from dryml.core2.object import Remember

    def _build_definition_enter(path, key, value):
        id_value = id(value)
        if id_value in instance_cache:
            return value, False
        elif isinstance(value, Remember):
            return {}, get_remember_view(value)
        elif isinstance(value, Definition):
            # We can encounter definitions we have already created
            found_def = False
            for _, v in instance_cache.items():
                if id(v) == id(value):
                    found_def = True
                    break
            if not found_def:
                raise ValueError("We should have built this defintion ourselves")
            else:
                # We don't want to enter already built definitions
                value, False
        else:
            return default_enter(path, key, value)

    def _build_definition_visit(_, key, value):
        id_value = id(value)
        if id_value in instance_cache:
            # First return any instance we have already cached
            return key, instance_cache[id_value]
        elif isinstance(value, Remember):
            raise TypeError("Unexpected type!")
        elif isinstance(value, Definition):
            # We should've stored this definition in the cache already
            found_def = False
            for _, v in instance_cache.items():
                if id(v) == id(value):
                    found_def = True
                    break
            if not found_def:
                raise TypeError("We should have constructed this Definition!")
            else:
                return key, value
        elif is_collection(value) or is_dictlike(value):
            # Don't do anything to the collections
            return key, value
        else:
            # This is a regular value. We need to deepcopy it.
            return key, deepcopy(value)

    def _build_definition_exit(path, key, values, new_parent, new_items):
        if isinstance(values, Remember) and type(new_parent) is dict:
            new_values = {}
            for k, v in new_items:
                new_values[k] = v
            args = new_values['args']
            kwargs = new_values['kwargs']
            # We want to copy the arguments so we don't
            # mutate them unless they're definitions or Objects
            args = deepcopy_skip_definition_object(args)
            kwargs = deepcopy_skip_definition_object(kwargs)

            new_def = Definition(
                type(values),
                *args,
                **kwargs)

            # Cache the instance result
            instance_cache[id(values)] = new_def

            return new_def
        else:
            return default_exit(path, key, values, new_parent, new_items)

    if isinstance(obj, Remember):
        return remap([obj], enter=_build_definition_enter, visit=_build_definition_visit, exit=_build_definition_exit)[0]
    else:
        return remap(obj, enter=_build_definition_enter, visit=_build_definition_visit, exit=_build_definition_exit)



# Creating objects from definitions
def build_from_definition(definition, path=None, repo=None):
    # First, concretize the definition
    concrete_definition = definition.concretize()

    # concrete definitions refer to specific objects

    with manage_repo(path=path, repo=repo) as repo:
        def build_from_definition_visit(_, key, value):
            if type(value) is Definition:
                raise TypeError("Definitions should've been turned into ConcreteDefinitions at this point")
            elif type(value) is ConcreteDefinition:
                # Delegate to a repo to do the loading
                obj = repo.load_object(value)
                return key, obj
            else:
                return key, value

        if isinstance(definition, Definition):
            return remap([concrete_definition], enter=definition_enter, visit=build_from_definition_visit, exit=definition_exit)[0]
        else:
            return remap(concrete_definition, enter=definition_enter, visit=build_from_definition_visit, exit=definition_exit)


def get_path(obj_or_def, path):
    from dryml.core2.object import Remember
    if len(path) == 0:
        return obj_or_def

    key = path[0]
    if key is None:
        return obj_or_def

    path = path[1:]
    if isinstance(obj_or_def, Remember):
        if key == 'cls':
            value = type(obj_or_def)
        elif key == 'args':
            value = obj_or_def.__args__
        elif key == 'kwargs':
            value = obj_or_def.__kwargs__
        else:
            raise KeyError(f"Invalid key {key} for Remember object")
    else:
        value = obj_or_def[key]

    try:
        return get_path(value, path)
    except (KeyError, IndexError) as e:
        raise PathAccessError(f"Can't access key path {path}")


## Selecting objects
def selector_match(selector, definition, strict=False):
    # Method for testing if a selector matches a definition
    # if strict is set, it must match exactly, and callables arent' allowed.
    # Additionally, Definitions which skip args also aren't allowed
    from dryml.core2.object import Remember

    print(f"selector_match: selector:")
    pprint(selector)
    print(f"selector_match: definition:")
    pprint(definition)

    def _selector_match_enter(path, key, value):
        print(f"_selector_match_enter: 1 path: {path} key: {key} value: {value}")
        if isinstance(value, Definition):
            print(f"_selector_match_enter: 2 definition branch")
            if strict and value.skip_args:
                raise TypeError("Definitions which skip args aren't allowed in strict mode")
            return {}, ItemsView(value)
        elif isinstance(value, Remember):
            print(f"_selector_match_enter: 3 remember branch")
            return {}, get_remember_view(value)
        else:
            print(f"_selector_match_enter: 4 default branch")
            return default_enter(path, key, value)

    def _selector_match_visit(path, key, value):
        # Try to get the value at the right path from the definition
        print(f"_selector_match_visit: 1 path: {path} key: {key} value: {value}")
        try:
            def_val = get_path(definition, path+(key,))
        except PathAccessError:
            return key, False
        print(f"_selector_match_visit: 2 def_val: {def_val}")

        if isclass(def_val):
            # We have a class in the definition.
            # If the selector value is a class, then the definition value must be a subclass.
            # This must also work for objects with metaclasses which aren't type
            if isclass(value):
                return key, issubclass(value, def_val)
            elif callable(value) and not strict:
                # We use the callable to determine if we match
                return key, value(def_val)
            elif isinstance(value, str):
                # We can do a class string comparison
                return key, value == get_class_str(def_val)
            else:
                # we don't have the right type in the selector
                return key, False
        elif (is_collection(def_val) or is_dictlike(def_val)) and not isinstance(def_val, np.ndarray):
            # We do nothing for these collections. Wait for their elements to be matched
            return key, value
        elif isinstance(def_val, np.ndarray):
            if isinstance(value, np.ndarray):
                return key, np.all(def_val == value)
            elif is_nonclass_callable(value) and not strict:
                return key, value(def_val)
            else:
                # type doesn't match.
                return key, False
        elif (type(value) is bool) and isinstance(def_val, Remember):
            # We've come out of a Remember object comparison.
            return key, value
        else:
            # Plain matching branch
            if is_nonclass_callable(value) and not strict:
                return key, value(def_val)
            elif type(value) != type(value):
                return key, False
            else:
                return key, value == def_val
            

    def _selector_match_exit(path, key, old_parent, new_parent, new_items):
        print(f"_selector_match_exit: 1 path: {path} key: {key} old_parent: {old_parent} new_parent: {new_parent} new_items: {new_items}")
        # Type check
        if type(old_parent) != type(new_parent):
            if isinstance(old_parent, Definition) or isinstance(old_parent, Remember):
                if type(new_parent) is not dict:
                    # The one case we know about should have new_parent be a dict.
                    return False
            else:
                return False

        def_values = get_path(definition, path+(key,))

        if strict:
            if len(def_values) != len(new_items):
                return False

        if is_collection(old_parent) and not is_dictlike(old_parent):
            # For tuples and list arguments, the lengths must match.
            if len(def_values) != len(new_items):
                return False
        final = True
        for val in map(lambda t: t[1], new_items):
            final = final & val
        return final

    # We reduce across the selector because we are only checking the values supplied
    # In the selector.
    return remap(selector, enter=_selector_match_enter, visit=_selector_match_visit, exit=_selector_match_exit)
