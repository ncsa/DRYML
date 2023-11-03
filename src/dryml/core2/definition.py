from copy import deepcopy
from boltons.iterutils import remap, is_collection, PathAccessError, default_enter, default_exit
import hashlib
from inspect import isclass
import numpy as np
import sys

from dryml.core2.util import is_dictlike, cls_super, \
    get_class_str, is_nonclass_callable, hashval_to_digest, \
    digest_to_hashval, get_remember_view, get_definition_view
from dryml.core2.repo import manage_repo


def definition_enter(path, key, value):
    if isinstance(value, Definition):
        return {}, get_definition_view(value)
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
    def _enter(path, key, value):
        from dryml.core2.object import Object
        if isinstance(value, Object):
            return value, False
        elif isinstance(value, Definition):
            return value, False
        else:
            return default_enter(path, key, value)

    def _visit(path, key, value):
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
        return remap(
            [defn],
            enter=_enter,
            visit=_visit,
            exit=definition_exit)[0]
    else:
        return remap(
            defn,
            enter=_enter,
            visit=_visit,
            exit=definition_exit)


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
        return deepcopy_skip_definition_object(self)

    def build(self, build_missing=True, **kwargs):
        return build_from_definition(self, build_missing=build_missing, **kwargs)

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

    def categorical(self, recursive=False):
        return categorical_definition(self, recursive=recursive)

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


def categorical_definition(defn: Definition, recursive=True):
    from dryml.core2.object import Remember
    # Copy the Definition
    new_def = deepcopy_skip_definition_object(defn)

    definition_cache = {}
    level = 0

    def _enter(path, key, value):
        if id(value) in definition_cache:
            return value, False
        elif isinstance(value, Remember):
            raise TypeError("Plain remember objects are not supported")
        elif isinstance(value, Definition):
            nonlocal level
            level += 1
            return {}, get_definition_view(value)
        else:
            return default_enter(path, key, value)

    def _visit(path, key, value):
        if isinstance(value, ConcreteDefinition):
            # We shouldn't have any ConcreteDefinitions at this point
            raise TypeError("ConcreteDefinition should not be here at this point")
        elif isinstance(value, Remember):
            raise TypeError("Plain Remember objects are not supported")
        else:
            return key, value

    def _exit(path, key, value, new_parent, new_items):
        if isinstance(value, Definition):
            # This should catch both Definitions and ConcreteDefinitions
            nonlocal level
            level -= 1
            new_vals = {}
            for k, v in new_items:
                new_vals[k] = v
            args = new_vals['args']
            kwargs = new_vals['kwargs']
            if not recursive:
                if level == 0:
                    # Only apply __strip_unique_args__ at the lowest level.
                    args, kwargs = cls_super(new_vals['cls']).__strip_unique_args__(*args, **kwargs)
            else:
                # Apply __strip_unique_args__ at all levels.
                cls_s = cls_super(new_vals['cls'])
                args, kwargs = cls_s.__strip_unique_args__(*args, **kwargs)
            return Definition(new_vals['cls'], *args, **kwargs)
        else:
            return default_exit(path, key, value, new_parent, new_items)

    if isinstance(new_def, Definition):
        return remap(
            [new_def],
            enter=_enter,
            visit=_visit,
            exit=_exit)[0]
    else:
        return remap(
            new_def,
            enter=_enter,
            visit=_visit,
            exit=_exit)


def concretize_definition(defn: Definition):
    from dryml.core2.object import Remember
    # Cache for completed results. All duplicate ConcreteDefinitions should refer to the SAME object and so the same definition
    # Key for this cache will be the ConcreteDefinition hash.
    definition_cache = {}
    built_definitions = {}

    def _enter(path, key, value):
        if id(value) in definition_cache:
            # We've seen this object before
            return value, False
        elif isinstance(value, ConcreteDefinition):
            # The definition is already concrete. don't enter it.
            return value, False
        elif isinstance(value, Definition):
            return {}, get_definition_view(value)
        elif isinstance(value, Remember):
            return {}, get_remember_view(value)
        else:
            return default_enter(path, key, value)


    def _visit(path, key, value):
        if id(value) in definition_cache: 
            # We've seen this object before
            return key, definition_cache[id(value)]
        elif type(value) is ConcreteDefinition:
            # Value is already Concrete
            return key, value
        elif isinstance(value, Remember):
            # We have an already realized class instance. We shouldn't deep copy it.
            raise TypeError("We shouldn't get a Remember object here.")
        elif isinstance(value, Definition):
            raise TypeError("We shouldn't get a Definition object here.")
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
        else:
            return key, deepcopy(value)

    def _create_def(new_parent, new_items):
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
        return ConcreteDefinition(cls, *args, **kwargs) 

    def _exit(path, key, values, new_parent, new_items):
        is_def = isinstance(values, Definition)
        is_rem = isinstance(values, Remember)
        if is_def or is_rem:
            if is_rem:
                # Check if we've seen this object's definition before.
                if values in built_definitions:
                    definition_cache[id(values)] = built_definitions[values]
                    return definition_cache[id(values)]

            new_def = _create_def(new_parent, new_items)
            built_definitions[new_def] = new_def
            # Check if we've encountered this definition before
            definition_cache[id(values)] = new_def
            return new_def
        else:
            return default_exit(path, key, values, new_parent, new_items)

    if isinstance(defn, Definition):
        return remap(
            [defn],
            enter=_enter,
            visit=_visit,
            exit=_exit)[0]
    else:
        return remap(
            defn,
            enter=_enter,
            visit=_visit,
            exit=_exit)


def validate_arguments_for_concrete_definition(vals):
    from dryml.core2.object import Remember
    # TODO: Maybe also directly validate for 'hashable' plain old data types as well?
    type_errors = []

    def _enter(path, key, value):
        if isinstance(value, ConcreteDefinition):
            # We assume any passed ConcreteDefinition object is already validated
            return key, False
        if isinstance(value, Definition):
            return {}, get_definition_view(value)
        elif isinstance(value, Remember):
            return {}, get_remember_view(value)
        else:
            return default_enter(path, key, value)

    def _visit(path, key, value):
        nonlocal type_errors
        if isinstance(value, ConcreteDefinition):
            pass
        elif isinstance(value, Definition):
            type_errors += [ ( path+(key,), type(value)) ]
        elif isinstance(value, Remember):
            type_errors += [ ( path+(key,), type(value)) ]
        return key, value

    def _exit(path, key, value, new_parent, new_items):
        # We aren't doing any transformation, so return the original
        return value

    remap(
        vals,
        enter=_enter,
        visit=_visit,
        exit=_exit)

    if len(type_errors) > 0:
        msg = ["The objects at the following paths are of disallowed types for ConcreteDefinition."]
        for t_error in type_errors:
            full_path = '/'.join(t_error[0])
            msg += [ f"{full_path}: class: {t_error[1].__name__}" ]
        raise TypeError('\n'.join(msg))


class ConcreteDefinition(Definition):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError("ConcreteDefinition must be created with arguments")
        if len(args) > 0:
            if not isclass(args[0]):
                raise TypeError("ConcreteDefinition's first argument must be a class")
        # Input validation
        validate_arguments_for_concrete_definition({'args': args, 'kwargs': kwargs})
        super().__init__(*args, **kwargs)
        # Pre-compute hash
        # TODO: pickling this object should not save this hash
        # We may decide later to change the hashing algorithm.
        self._hash = digest_to_hashval(hash_function(self))

    def concretize(self):
        return self

    def __hash__(self):
        return self._hash


def hash_value(value):
    # Hashes for supported values here
    if isclass(value):
        return str(get_class_str(value))

    try:
        hash_val = hash(value)
        return hashval_to_digest(hash_val)
    except TypeError:
        pass

    if isinstance(value, np.ndarray):
        # Hash for numpy arrays
        return hashlib.sha256(value.tobytes()).hexdigest()
    else:
        raise TypeError(f"Value of type {type(value)} not supported for hashing.")


def hash_function(structure):
    # Definition hash support
    class HashHelper(object):
        def __init__(self, the_hash):
            self.hash = the_hash

    def _visit(path, key, value):
        # Skip if it's a hashlib hasher
        if isinstance(value, HashHelper):
            return key, value.hash
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
    
        return key, hash_value(value)

    def _exit(path, key, old_parent, new_parent, new_items):
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

    return remap(
        structure,
        enter=definition_enter,
        visit=_visit,
        exit=_exit).hash


# Creating definitions from objects
def build_definition(obj):
    instance_cache = {}
    from dryml.core2.object import Remember

    def _enter(path, key, value):
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

    def _visit(_, key, value):
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

    def _exit(path, key, values, new_parent, new_items):
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
        return remap(
            [obj],
            enter=_enter,
            visit=_visit,
            exit=_exit)[0]
    else:
        return remap(
            obj,
            enter=_enter,
            visit=_visit,
            exit=_exit)



# Creating objects from definitions
def build_from_definition(definition, build_missing=True, **kwargs):
    with manage_repo(**kwargs) as repo:
        # Get unique objects
        unique_objs = unique_remember_objects(definition)

        # Add these objects to the repo.
        if len(unique_objs) > 0:
            repo.add_object(*(unique_objs.values()))

        # First, concretize the definition
        concrete_definition = concretize_definition(definition)
        # concrete definitions refer to specific objects

        def _visit(path, key, value):
            # Do nothing on visit.
            return key, value

        def _exit(path, key, value, new_parent, new_items):
            if isinstance(value, ConcreteDefinition):
                return repo.load_object(value, build_missing=build_missing)
            else:
                return default_exit(path, key, value, new_parent, new_items)

        if isinstance(definition, Definition):
            return remap(
                [concrete_definition],
                enter=definition_enter,
                visit=_visit,
                exit=_exit)[0]
        else:
            return remap(
                concrete_definition,
                enter=definition_enter,
                visit=_visit,
                exit=_exit)


def get_path(obj_or_def, path):
    from dryml.core2.object import Remember
    if len(path) == 0:
        return obj_or_def

    key = path[0]
    if key is None:
        return obj_or_def

    new_path = path[1:]
    if isinstance(obj_or_def, Remember):
        if key == 'cls':
            value = type(obj_or_def)
        elif key == 'args':
            value = obj_or_def.__args__
        elif key == 'kwargs':
            value = obj_or_def.__kwargs__
        else:
            raise PathAccessError(KeyError("Unsupported key on Remember Object"), key, new_path)
    else:
        try:
            value = obj_or_def[key]
        except (KeyError, IndexError, ValueError, TypeError, PathAccessError) as e:
            if type(e) is PathAccessError:
                raise PathAccessError(e.exc, key, new_path)
            else:
                raise PathAccessError(e, key, new_path)


    try:
        return get_path(value, new_path)
    except (KeyError, IndexError, ValueError, PathAccessError) as e:
        if type(e) is PathAccessError:
            raise PathAccessError(e.exc, key, new_path)
        else:
            raise PathAccessError(e, key, new_path)


def render_path(path, key):
    path = ("root",) + path
    if key is not None:
        path = path + (key,)

    return "/".join(map(str, path))


## Selecting objects
def selector_match(selector, definition, strict=False, cls_str_compare=False, verbose=False, output_stream=sys.stderr):
    # Method for testing if a selector matches a definition
    # if strict is set, it must match exactly, and callables arent' allowed.
    # cls_str_compare forces a string based name comparison between classes.
    # Additionally, Definitions which skip args also aren't allowed
    from dryml.core2.object import Remember

    def _enter(path, key, value):
        # Check if this key/path exists in the definition
        try:
            def_val = get_path(definition, path+(key,))
        except PathAccessError:
            if verbose:
                print(
                        f"[{render_path(path, key)}]: Doesn't exist in target\n")
            return key, False
        if isinstance(value, Definition):
            if strict and value.skip_args:
                raise TypeError("Definitions which skip args aren't allowed in strict mode")
            return {}, get_definition_view(value)
        elif isinstance(value, Remember):
            return {}, get_remember_view(value)
        else:
            return default_enter(path, key, value)

    def _visit(path, key, value):
        # Try to get the value at the right path from the definition
        # Grab the definition value
        try:
            def_val = get_path(definition, path+(key,))
        except PathAccessError:
            return key, False

        if isclass(def_val):
            # We have a class in the definition.
            # If the selector value is a class, then the definition value must be a subclass.
            # This must also work for objects with metaclasses which aren't type
            if isclass(value):
                if strict:
                    condition = value is def_val
                    if not condition and verbose:
                        print(
                            f"[{render_path(path, key)}]: Classes differ\n",
                            file=output_stream)
                    return key, condition
                else:
                    condition = issubclass(def_val, value)
                    if not condition and verbose:
                        print(
                            f"[{render_path(path, key)}]: {get_class_str(def_val)} is not a subclass of {get_class_str(value)}\n",
                            file=output_stream)
                    return key, condition
            elif callable(value) and not strict:
                # We use the callable to determine if we match
                condition = value(def_val)
                if not condition and verbose:
                    print(
                        f"[{render_path(path, key)}]: Callable test failed\n",
                        file=output_stream)
                return key, condition
            elif isinstance(value, str):
                # We can do a class string comparison
                condition = (value == get_class_str(def_val))
                if not condition and verbose:
                    print(
                        f"[{render_path(path, key)}]: {value} failed string based class comparison\n",
                        file=output_stream)
                return key, condition
            else:
                if verbose:
                    print(
                        f"[{render_path(path, key)}]: type {type(value)} unsupported for class comparison\n",
                        file=output_stream)
                # we don't have the right type in the selector
                return key, False
        elif (is_collection(def_val) or is_dictlike(def_val)) and not isinstance(def_val, np.ndarray):
            # We do some type checking on this visit. deeper structures have already been visited
            sel_val = get_path(selector, path+(key,))
            if type(sel_val) is not type(def_val):
                # Container class doesn't match
                if verbose:
                    print(
                        f"[{render_path(path, key)}]: Classes don't match. {type(sel_val)} in the selector {type(def_val)} in the target\n",
                        file=output_stream)
                return key, False
            return key, value
        elif isinstance(def_val, np.ndarray):
            if isinstance(value, np.ndarray):
                condition = (value.shape == def_val.shape)
                if not condition:
                    if verbose:
                        print(
                            f"[{render_path(path, key)}]: Mismatched array shapes {value.shape} != {def_val.shape}\n",
                            file=output_stream)
                    return key, condition
                condition = np.all(def_val == value)
                if not condition and verbose:
                    print(
                        f"[{render_path(path, key)}]: Unequal Arrays\n",
                        file=output_stream)
                return key, condition
            elif is_nonclass_callable(value) and not strict:
                condition = value(def_val)
                if not condition and verbose:
                    print(
                        f"[{render_path(path, key)}]: Callable test failed\n",
                        file=output_stream)
                return key, condition
            else:
                if not condition and verbose:
                    print(
                        f"[{render_path(path, key)}]: Selector value type {type(value)} wrong for array comparison\n",
                        file=output_stream)
                # type doesn't match.
                return key, False
        elif (type(value) is bool) and isinstance(def_val, Remember):
            # We've come out of a Remember object comparison.
            return key, value
        else:
            # Plain matching branch
            if is_nonclass_callable(value) and not strict:
                condition = value(def_val)
                if not condition and verbose:
                    print(
                        f"[{render_path(path, key)}]: Callable test failed\n",
                        file=output_stream)
                return key, condition
            elif type(value) is not type(def_val):
                if verbose:
                    print(
                        f"[{render_path(path, key)}]: Type mismatch\n",
                        file=output_stream)
                return key, False
            else:
                condition = (value == def_val)
                if not condition and verbose:
                    print(
                        f"[{render_path(path, key)}]: Values differ\n",
                        file=output_stream)
                return key, condition
            

    def _exit(path, key, old_parent, new_parent, new_items):
        # Type check
        if type(old_parent) != type(new_parent):
            if isinstance(old_parent, Definition) or isinstance(old_parent, Remember):
                if type(new_parent) is not dict:
                    # The one case we know about should have new_parent be a dict.
                    return False
            else:
                return False

        # This should also not throw any errors.
        def_values = get_path(definition, path+(key,))

        # We expect to be comparing collections at this point.
        if not isinstance(def_values, Remember) and \
                (not is_collection(def_values) and not is_dictlike(def_values)):
            return False

        # Here, we compute the number of values there should be with a special case for 'single' Remember values.
        if isinstance(def_values, Remember):
            num_def_values = 1 + len(def_values.__args__) + len(def_values.__kwargs__)
        else:
            num_def_values = len(def_values)

        if strict:
            if num_def_values != len(new_items):
                return False

        if is_collection(old_parent) and not is_dictlike(old_parent):
            # For tuples and list arguments, the lengths must match.
            if num_def_values != len(new_items):
                return False
        final = True
        for val in map(lambda t: t[1], new_items):
            final = final & val
        return final

    # We reduce across the selector because we are only checking the values supplied
    # In the selector.
    return remap(
        selector,
        enter=_enter,
        visit=_visit,
        exit=_exit)


def unique_remember_objects(def_or_obj):
    # Get a dictionary of unique Remember objects inside
    # the nested definition or Remember object.
    from dryml.core2.object import Remember
    unique_objs = {}

    def _enter(path, key, value):
        if isinstance(value, Remember):
            return {}, get_remember_view(value)
        elif isinstance(value, Definition):
            return {}, get_definition_view(value)
        return default_enter(path, key, value)

    def _visit(path, key, value):
        return key, value

    def _exit(path, key, value, new_parent, new_items):
        if isinstance(value, Remember):
            # Add the remember object to our dictionary
            # if we haven't seen it yet.
            if value not in unique_objs:
                unique_objs[value] = value
        return key, None

    if isinstance(def_or_obj, Remember):
        remap(
            [def_or_obj],
            enter=_enter,
            visit=_visit,
            exit=_exit)
    else:
        remap(
            def_or_obj,
            enter=_enter,
            visit=_visit,
            exit=_exit)

    return unique_objs
