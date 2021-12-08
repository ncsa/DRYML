import os
from dryml.dry_object import DryObject, DryObjectFactory, DryObjectFile
from dryml.dry_selector import DrySelector
from typing import Optional
import enum
import tqdm

# This type will act as a fascade for the various DryObject* types.
class DryRepo(object):
    def __init__(self, directory:Optional[str]=None, create:bool=False, **kwargs):
        super().__init__(**kwargs)

        if directory is not None: 
            self.link_to_directory(directory)

        # A dictionary of objects
        self.obj_dict = {}

        # Model list (for iteration)
        self.obj_list = []

    def link_to_directory(self, directory:str, create:bool=False):
        # Check that directory exists
        if not os.path.exists(directory):
            if create:
                os.mkdir(directory)
        self.directory = directory

    def number_of_objects(self):
        return len(self.obj_list)

    def load_objects_from_directory(self, selector:Optional[DrySelector]=None, verbose:bool=False):
        "Method to refresh the internal dictionary of objects."
        if self.directory is None:
            # Don't do anything if no directory is linked
            return
        files = os.listdir(self.directory)

        num_loaded = 0
        for filepath in files:
            with DryObjectFile(filepath) as f:
                if selector is not None:
                    if not selector(f):
                        # Skip non selected objects
                        continue
                obj_cat_hash = f.get_category_hash()
                obj_hash = f.get_individual_hash()
                if obj_cat_hash not in self.obj_dict:
                    self.obj_dict[obj_cat_hash] = {}
                if obj_hash not in self.obj_dict[obj_cat_hash]:
                    num_loaded += 1
                    obj_definition = {
                        'val': filepath,
                        'filepath': filepath
                    }
                    self.obj_dict[obj_cat_hash][obj_hash] = obj_definition
                    self.obj_list.append(obj_definition)
        if verbose:
            print(f"Loaded {num_loaded} objects")

    def add_object(self, obj: DryObject, filename:Optional[str]=None):
        # Add a single object
        obj_cat_hash = obj.get_category_hash()
        obj_hash = obj.get_individual_hash()
        obj_definition = {'val': obj}
        if obj_cat_hash not in self.obj_dict:
            self.obj_dict[obj_cat_hash] = {}
        if obj_hash in self.obj_dict[obj_cat_hash]:
            raise ValueError("Object {obj_hash} in category {obj_cat_hash} already exists in the repo!")
        self.obj_dict[obj_cat_hash][obj_hash] = obj_definition
        self.obj_list.append(obj_definition)

    def add_objects(self, obj_factory: DryObjectFactory, num=1):
        # Create numerous objects from a factory function
        for i in range(num):
            obj = obj_factory()
            self.add_object(obj)

    @staticmethod
    def make_filter_func(selector:DrySelector, sel_args=None, sel_kwargs=None):
        if sel_args is None:
            sel_args = []
        if sel_kwargs is None:
            sel_kwargs = {}
        def filter_func(obj_container):
            if isinstance(obj_container['val'], str):
                # This container just a string, we need to load from disk to check against the selector
                with DryObjectFile(filepath) as f:
                    if selector(f, *sel_args, **sel_kwargs):
                        # Load the objects, as we will need them
                        obj_container['val'] = f.load_object(update=update)
                        return True
                    else:
                    	return False
            elif isinstance(obj_container['val'], DryObject):
                if selector(obj_container['val'], *sel_args, **sel_kwargs):
                    return True
                else:
                    return False
            elif isinstance(obj_container['val'], DryObjectFile):
                f = obj_container['val']
                if selector(obj_container['val'], *sel_args, **sel_kwargs):
                    obj_container['val'] = f.load_object(update=update)
                    return True
                else:
                    return False
        return filter_func

    def get_objs(self, selector:Optional[DrySelector]=None, sel_args=None, sel_kwargs=None):
        if selector is not None:
            filter_func = DryRepo.make_filter_func(selector, sel_args=sel_args, sel_kwargs=sel_kwargs)
            obj_list = list(filter(filter_func, self.obj_list))
        else:
            obj_list = self.obj_list

        return list(map(lambda o: o['val'], obj_list))

    def apply_to_objs(self, func, func_args=None, func_kwargs=None, selector:Optional[DrySelector]=None,
                      update:bool=True, verbosity:int=0, sel_args=None, sel_kwargs=None):
        "Apply a function to all objects tracked by the repo. We can also use a DrySelector to apply only to specific models"
        def apply_func(obj_container):
            obj = obj_container['val']
            if not isinstance(obj, DryObject):
                raise TypeError("Only objects of type DryObject are supported!")
            if verbosity == 1:
                cat_hash = obj.get_category_hash()
                ind_hash = obj.get_individual_hash()
                print(f"applying to model {ind_hash} of category {cat_hash}")
            if verbosity >= 2:
                ind_hash = obj.get_individual_hash()
                cat_hash = obj.get_hash_str()
                print(f"apply to model {ind_hash} of category {cat_hash}")
            return func(obj, *func_args, **func_kwargs)

        if selector is not None:
            filter_func = DryRepo.make_filter_func(selector, sel_args=sel_args, sel_kwargs=sel_kwargs)
            obj_list = list(filter(filter_func, self.obj_list))
        else:
            obj_list = self.obj_list

        num_objects = len(obj_list)

        # apply function to objects
        results = list(map(apply_func, tqdm(obj_list)))

        if len(list(filter(lambda x: x is None, results))) == num_objects:
            return None
        else:
            return results
