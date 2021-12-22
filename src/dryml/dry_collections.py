from collections import UserList
from dryml.dry_object import DryObject, DryObjectDefinition
from dryml.utils import init_arg_dict_handler, init_arg_list_handler, \
    is_dictlike


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
                # Add definition dictionary to dry_args
                dry_args.append(arg.get_definition().to_dict())
                # Append the object to the list of objects
                objs.append(arg)
            elif is_dictlike(arg):
                # Create definition from dictlike argument
                obj_def = DryObjectDefinition.from_dict(arg)
                # Create the object and add it to the list
                obj = DryObjectDefinition.from_dict(arg)()
                objs.append(obj)
                # Append DryObjectDefinition dict to the arguments
                dry_args.append(obj_def.to_dict())
            else:
                raise ValueError(f"Unsupported argument type: {arg}")

        super().__init__(
            dry_args=dry_args,
            dry_kwargs=dry_kwargs,
            **kwargs)

        for obj in objs:
            self.append(obj)
