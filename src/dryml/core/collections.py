from collections import UserList, UserDict
from dryml.object import Object, ObjectDef
from dryml.config import Meta
from typing import Mapping


class List(Object, UserList):
    @Meta.collect_args
    def __init__(self, *args, **kwargs):
        objs = []
        for arg in args:
            if not isinstance(arg, Object):
                raise ValueError(
                    "List does not support elements of type"
                    f" {type(arg)}.")
            else:
                objs.append(arg)

        self.data.extend(objs)

    # We have to do a special implementation of definition
    # We want the reported dry_args to always match whats in
    # the list. this should be computed dynamically
    def definition(self):
        dry_args = []
        for obj in self:
            dry_args.append(obj.definition())
        return ObjectDef(
            type(self),
            *dry_args,
            dry_mut=True,
            **self.dry_kwargs)


class Tuple(Object):
    @Meta.collect_args
    def __init__(self, *args, **kwargs):
        objs = []
        for obj in args:
            if not isinstance(obj, Object):
                raise ValueError(f"Unsupported element of type: {type(obj)}")
            else:
                objs.append(obj)
        self.data = tuple(objs)

    def __getitem__(self, key):
        # Accessor
        return self.data[key]

    def __len__(self):
        return len(self.data)

    # We have to do a special implementation of definition
    # We want the reported dry_args to always match whats in
    # the list. this should be computed dynamically
    def definition(self):
        dry_args = []
        is_mutable = False
        for obj in self.data:
            obj_def = obj.definition()
            if obj_def.dry_mut:
                is_mutable = True
            dry_args.append(obj_def)
        return ObjectDef(
            type(self),
            *dry_args,
            dry_mut=is_mutable,
            **self.dry_kwargs)


class Dict(Object, UserDict):
    def __init__(
            self, in_dict: Mapping, **kwargs):
        for key in in_dict:
            self.data[key] = in_dict[key]

    # We have to do a special implementation of definition
    # We want the reported dry_args to always match whats in
    # the list. this should be computed dynamically
    def definition(self):
        # Build dry arg dictionary
        dry_arg = {}
        for key in self:
            obj = self[key]
            dry_arg[key] = obj.definition()
        return ObjectDef(
            type(self),
            dry_arg,
            dry_mut=True,
            **self.dry_kwargs)
