import os
from dryml.dry_object import DryObject, DryObjectFactory, DryObjectFile, change_object_cls
from dryml.dry_selector import DrySelector
from typing import Optional, Callable
import io
import enum
import tqdm

# This type will act as a fascade for the various DryObject* types.
class DryRepo(object):
    def __init__(self, directory:Optional[str]=None, create:bool=False, load_objects:bool=True, **kwargs):
        super().__init__(**kwargs)

        # A dictionary of objects
        self.obj_dict = {}

        # Model list (for iteration)
        self.obj_list = []

        if directory is not None: 
            self.link_to_directory(directory, create=create)
            if load_objects:
                self.load_objects_from_directory()

    def link_to_directory(self, directory:str, create:bool=False):
        # Check that directory exists
        if not os.path.exists(directory):
            if create:
                os.mkdir(directory)
        self.directory = directory

    def number_of_objects(self):
        return len(self.obj_list)

    def load_objects_from_directory(self, selector:Optional[Callable]=None, verbose:bool=False):
        "Method to refresh the internal dictionary of objects."
        if self.directory is None:
            # Don't do anything if no directory is linked
            return
        files = os.listdir(self.directory)

        num_loaded = 0
        for filepath in files:
            full_filepath = os.path.join(self.directory, filepath)
            try:
                with DryObjectFile(full_filepath) as f:
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
            except Exception as e:
                print(f"WARNING! Malformed file found! {full_filepath} skipping load Error: {e}")
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

    def make_container_handler(self,
            load_object:bool=True,
            open_container:bool=True, update:bool=True):
        def container_handler(obj_container):
            def container_opener(obj_container):
                if open_container:
                    return obj_container['val']
                else:
                    return obj_container
            if isinstance(obj_container['val'], str):
                if load_object:
                    if self.directory is None:
                        raise RuntimeError("Repo is not linked to a directory!")
                    filepath = os.path.join(self.directory, obj_container['filepath'])
                    with DryObjectFile(filepath) as f:
                        obj_container['val'] = f.load_object(update=update)
                return container_opener(obj_container)
            elif isinstance(obj_container['val'], DryObjectFile):
                if load_object:
                    f = obj_container['val']
                    obj_container['val'] = f.load_object(update=update)
                    f.close()
                return container_opener(obj_container)
            elif isinstance(obj_container['val'], DryObject):
                return container_opener(obj_container)
            else:
                raise RuntimeError(f"Unsupported value type: {type(obj_container['val'])}")

        return container_handler

    def make_filter_func(self,
            selector:Optional[Callable]=None, sel_args=None, sel_kwargs=None,
            only_objs:bool=False):
        if sel_args is None:
            sel_args = []
        if sel_kwargs is None:
            sel_kwargs = {}
        def filter_func(obj_container):
            if isinstance(obj_container['val'], str):
                if only_objs:
                    return False
                if self.directory is None:
                    raise RuntimeError("Repo is not connected to any directory!")
                # This container just a string, we need to load from disk to check against the selector
                filepath = os.path.join(self.directory, obj_container['val'])
                if selector is not None:
                    with DryObjectFile(filepath) as f:
                        if selector(f, *sel_args, **sel_kwargs):
                            return True
                        else:
                    	    return False
                else:
                    return True
            elif isinstance(obj_container['val'], DryObject):
                if selector is not None:
                    if selector(obj_container['val'], *sel_args, **sel_kwargs):
                        return True
                    else:
                        return False
                else:
                    return True
            elif isinstance(obj_container['val'], DryObjectFile):
                if only_objs:
                    return False
                if selector is not None:
                    if selector(obj_container['val'], *sel_args, **sel_kwargs):
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                raise RuntimeError(f"Unsupported value type: {type(obj_container['val'])}")
        return filter_func

    def get(self,
            selector:Optional[Callable]=None, sel_args=None, sel_kwargs=None,
            load_objects:bool=True, update:bool=True, open_container:bool=True,
            verbose:bool=True):
        # Handle arguments for filter function
        only_objs = True
        if load_objects:
            only_objs = False

        # Filter the internal object list
        filter_func = \
            self.make_filter_func(
                selector,
                sel_args=sel_args,
                sel_kwargs=sel_kwargs,
                only_objs=only_objs)
        obj_list = list(filter(filter_func, self.obj_list))

        # Build container handler
        container_handler = \
        	self.make_container_handler(
                update=update,
                open_container=open_container)
        return list(map(container_handler, obj_list))

    def apply(self,
            func, func_args=None, func_kwargs=None,
            selector:Optional[Callable]=None, sel_args=None, sel_kwargs=None,
            update:bool=True, verbose:int=0, load_objects:bool=True, open_container:bool=True):
        "Apply a function to all objects tracked by the repo. We can also use a DrySelector to apply only to specific models"
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = {}

        # Create apply function
        def apply_func(obj):
            return func(obj, *func_args, **func_kwargs)

        # Get object list
        objs = self.get(
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            update=update, load_objects=load_objects, open_container=open_container)

        # apply function to objects
        if verbose:
            results = list(map(apply_func, tqdm(objs)))
        else:
            results = list(map(apply_func, objs))

        # Handle results
        if len(list(filter(lambda x: x is None, results))) == len(objs):
            return None
        else:
            return results

    def reload_objs(self,
            selector:Optional[Callable]=None, sel_args=None, sel_kwargs=None,
            update:bool=False, reload:bool=False):
        if self.directory is None:
            raise RuntimeError("Repo directory needs to be set for reloading.")

        def reload_func(obj_container):
            obj = obj_container['val']
            if not isinstance(obj, DryObject):
                raise RuntimeError("Can only reload DryObjects")
            # Get current definition of class
            new_cls = get_current_cls(type(obj), reload=reload)
            del obj
            obj_container['val'] = change_object_cls(obj, new_cls)

        self.apply(reload_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, load_objects=False)


    def save(self,
            selector:Optional[Callable]=None, sel_args=None, sel_kwargs=None):
        if self.directory is None:
            raise RuntimeError("Repo's directory is not set. Set the directory")

        def save_func(obj_container):
            obj = obj_container['val']
            if not isinstance(obj, DryObject):
                raise RuntimeError("Can only save currently loaded DryObject")
            if 'filename' not in obj_container:
                obj_container['filename'] = str(obj.get_individual_hash())
            filename = os.path.join(self.directory, obj_container['filename'])
            obj.save_self(filename)

        self.apply(save_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False)
