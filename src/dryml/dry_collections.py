import zipfile
import pickle
from collections import UserList
from dryml.dry_object import DryObject, DryObjectDef, load_object
from dryml.utils import init_arg_dict_handler, init_arg_list_handler, \
    is_dictlike, pickler


class DryList(DryObject, UserList):
    def __init__(
            self, *args, dry_args=None,
            dry_kwargs=None, **kwargs):
        # Ingest Dry Args/Kwargs
        dry_args = init_arg_list_handler(dry_args)
        dry_kwargs = init_arg_dict_handler(dry_kwargs)

        objs = []

        for arg in args:
            if isinstance(arg, DryObject):
                # The list is given a DryObject directly.
                # We don't need to consult any repo.
                # Add definition dictionary to dry_args
                dry_args.append(arg.definition().to_dict())
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
                dry_args.append(obj_def.to_dict())
            else:
                raise ValueError(f"Unsupported argument type: {arg}")

        super().__init__(
            dry_args=dry_args,
            dry_kwargs=dry_kwargs,
            **kwargs)

        for obj in objs:
            self.append(obj)

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
