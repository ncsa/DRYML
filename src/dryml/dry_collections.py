from collections import UserList, UserDict
from dryml.dry_object import DryObject, DryObjectDef
from dryml.dry_config import DryMeta
from typing import Mapping


class DryList(DryObject, UserList):
    @DryMeta.collect_args
    def __init__(self, *args, **kwargs):
        objs = []
        for arg in args:
            if not isinstance(arg, DryObject):
                raise ValueError(
                    "Dry List does not support elements of type"
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
        return DryObjectDef(
            type(self),
            *dry_args,
            dry_mut=True,
            **self.dry_kwargs)


class DryTuple(DryObject):
    @DryMeta.collect_args
    def __init__(self, *args, **kwargs):
        objs = []
        for obj in args:
            if not isinstance(obj, DryObject):
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
        return DryObjectDef(
            type(self),
            *dry_args,
            dry_mut=is_mutable,
            **self.dry_kwargs)


class DryDict(DryObject, UserDict):
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
        return DryObjectDef(
            type(self),
            dry_arg,
            dry_mut=True,
            **self.dry_kwargs)
