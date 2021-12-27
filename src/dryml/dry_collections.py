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

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load super classes information
        if not super().load_object_imp(file):
            return False

        # Load component list
        with file.open('component_list.pkl', mode='r') as f:
            component_filenames = pickle.loads(f.read())

        # Load individual components
        for filename in component_filenames:
            with file.open(filename, mode='r') as f:
                self.components.append(load_object(f))

        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        # We save each component inside the file first.
        component_filenames = []
        for component in self.components:
            filename = f"{component.get_individual_hash()}.dry"
            with file.open(filename, mode='w') as f:
                component.save_self(f)
            component_filenames.append(filename)

        with file.open('component_list.pkl', mode='w') as f:
            f.write(pickler(component_filenames))

        # Super classes should save their information
        return super().save_object_imp(file)
