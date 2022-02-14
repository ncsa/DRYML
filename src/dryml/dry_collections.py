import zipfile
import pickle
from collections import UserList, UserDict
from dryml.dry_object import DryObject, DryObjectDef, load_object
from dryml.dry_config import DryMeta
from dryml.utils import pickler
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

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load object list
        with file.open('obj_list.pkl', mode='r') as f:
            obj_filenames = pickle.loads(f.read())

        if len(self) != len(obj_filenames):
            # Didn't load as many objects as saved filenames
            return False

        # Unload existing objects from the list
        self.clear()

        # Load objects
        for filename in obj_filenames:
            with file.open(filename, mode='r') as f:
                self.append(load_object(f))

        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        obj_filenames = []

        # We save each object inside the file first.
        for obj in self:
            filename = f"{obj.definition().get_individual_id()}.dry"
            with file.open(filename, mode='w') as f:
                obj.save_self(f)
            obj_filenames.append(filename)

        # Save object list
        with file.open('obj_list.pkl', mode='w') as f:
            f.write(pickler(obj_filenames))

        return True


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

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load object list
        with file.open('obj_list.pkl', mode='r') as f:
            obj_filenames = pickle.loads(f.read())

        if len(self) != len(obj_filenames):
            # Didn't load as many objects as saved filenames
            return False

        # Replace existing objects in the tuple
        new_tuple = []
        for i in range(len(obj_filenames)):
            with file.open(obj_filenames[i], mode='r') as f:
                new_tuple.append(load_object(f))
        self.data = tuple(new_tuple)

        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        obj_filenames = []

        # We save each object inside the file first.
        for obj in self:
            filename = f"{obj.definition().get_individual_id()}.dry"
            with file.open(filename, mode='w') as f:
                obj.save_self(f)
            obj_filenames.append(filename)

        # Save object list
        with file.open('obj_list.pkl', mode='w') as f:
            f.write(pickler(obj_filenames))

        return True


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

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load object list
        with file.open('obj_dict.pkl', mode='r') as f:
            obj_dict = pickle.loads(f.read())

        if len(self) != len(obj_dict):
            # Didn't load as many objects as saved filenames
            return False

        # Load objects
        for key in obj_dict:
            filename = obj_dict[key]
            with file.open(filename, mode='r') as f:
                self[key] = load_object(f)

        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        obj_dict = {}

        # We save each object inside the file first.
        for key in self:
            obj = self[key]
            filename = f"{obj.definition().get_individual_id()}.dry"
            with file.open(filename, mode='w') as f:
                obj.save_self(f)
            obj_dict[key] = filename

        # Save object list
        with file.open('obj_dict.pkl', mode='w') as f:
            f.write(pickler(obj_dict))

        return True
