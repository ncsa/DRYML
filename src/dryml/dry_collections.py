import zipfile
import pickle
from collections import UserList, UserDict
from dryml.dry_object import DryObject, DryObjectDef, load_object
from dryml.dry_config import DryMeta
from dryml.utils import is_dictlike, pickler


class DryList(DryObject, UserList):
    @DryMeta.skip_args
    def __init__(self, *args, **kwargs):
        objs = []

        for arg in args:
            if isinstance(arg, DryObject):
                # The list is given a DryObject directly.
                # We don't need to consult any repo.
                # Add definition dictionary to dry_args
                self.dry_args.append(arg.definition().to_dict())
                # Append the object to the list of objects
                objs.append(arg)
            elif is_dictlike(arg):
                # Create definition from dictlike argument
                # This means we might need to look in a repo
                obj_def = DryObjectDef.from_dict(arg)
                # Create the object and add it to the list
                obj = obj_def.build()
                objs.append(obj)
                # Append DryObjectDef dict to the arguments
                self.dry_args.append(obj_def.to_dict())
            else:
                raise ValueError(
                    f"Unsupported argument type: {type(arg)} - {arg}")

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
        # Load super classes information
        if not super().load_object_imp(file):
            return False

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

        # Super classes should save their information
        return super().save_object_imp(file)


class DryTuple(DryObject):
    @DryMeta.skip_args
    def __init__(self, *args, **kwargs):
        objs = []

        for arg in args:
            if isinstance(arg, DryObject):
                # The list is given a DryObject directly.
                # We don't need to consult any repo.
                # Add definition dictionary to dry_args
                self.dry_args.append(arg.definition().to_dict())
                # Append the object to the list of objects
                objs.append(arg)
            elif is_dictlike(arg):
                # Create definition from dictlike argument
                # This means we might need to look in a repo
                obj_def = DryObjectDef.from_dict(arg)
                # Create the object and add it to the list
                obj = obj_def.build()
                objs.append(obj)
                # Append DryObjectDef dict to the arguments
                self.dry_args.append(obj_def.to_dict())
            else:
                raise ValueError(f"Unsupported argument type: {arg}")

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
        # Load super classes information
        if not super().load_object_imp(file):
            return False

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

        # Super classes should save their information
        return super().save_object_imp(file)


class DryDict(DryObject, UserDict):
    @staticmethod
    def args_preprocess(obj, *args, **kwargs):
        if len(args) == 0:
            # WARNING! This could cause a race condition if python
            # becomes truely parallel!
            DryDict.__dry_data_temp__ = {}
            return ([], kwargs)

        if len(args) > 1:
            raise ValueError(
                "Extra positional arguments beyond the first are ignored "
                "for DryDict objects.")

        # get first argument
        in_dict = args[0]

        dry_arg = {}
        obj_dict = {}

        for key in in_dict:
            val = in_dict[key]
            if isinstance(val, DryObject):
                # The dict value is a DryObject directly.
                # We don't need to consult any repo.
                # Add definition dictionary to dry_arg
                dry_arg[key] = val.definition().to_dict()
                # Add the object to the dict
                obj_dict[key] = val
            elif is_dictlike(val):
                # Create DryObject from definition from dictlike argument
                # This means we might need to look in a repo
                obj_def = DryObjectDef.from_dict(val)
                # Add definition dict to the dry_arg
                dry_arg[key] = obj_def.to_dict()
                # Create the object and add it to the list
                obj = obj_def.build()
                obj_dict[key] = obj
            else:
                raise ValueError(f"Unsupported value type: {type(val)}")

        DryDict.__dry_data_temp__ = obj_dict

        return ([dry_arg], kwargs)

    def __init__(
            self, in_dict, **kwargs):
        # Copy data over from temp storage spot
        self.data = DryDict.__dry_data_temp__
        del DryDict.__dry_data_temp__

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
        # Load super classes information
        if not super().load_object_imp(file):
            return False

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

        # Super classes should save their information
        return super().save_object_imp(file)
