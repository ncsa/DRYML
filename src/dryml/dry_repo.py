import os
from dryml.dry_object import DryObject, DryObjectFactory, DryObjectFile, \
    DryObjectDef, change_object_cls, load_object
from dryml.utils import get_current_cls
from typing import Optional, Callable, Union
import tqdm
from pprint import pprint


class DryRepoContainer(object):
    @staticmethod
    def from_filepath(filename: str, directory: Optional[str] = None):
        new_container = DryRepoContainer(directory=directory)
        new_container._filename = filename
        return new_container

    @staticmethod
    def from_object(obj: DryObject, filename: Optional[str] = None,
                    directory: Optional[str] = None):
        new_container = DryRepoContainer(directory=directory)
        new_container._obj = obj
        if filename is not None:
            new_container._filename = filename
        else:
            # Set default filename
            new_container._filename = \
                obj.definition().get_individual_id()+'.dry'
        return new_container

    def __init__(self, directory: Optional[str] = None):
        self._directory = directory
        self._filename = None
        self._obj = None

    def __str__(self):
        if self._obj is None:
            obj_part = "Unloaded"
        else:
            obj_part = f"{self._obj}"

        try:
            path_part = f"{self.filepath}"
        except Exception:
            path_part = "No current filepath"

        def_part = str(self.definition())

        return f"DryRepoContainer: {obj_part} {def_part} at {path_part}"

    @property
    def filepath(self):
        if self._filename is None:
            raise RuntimeError(
                "No filename indicated "
                "Can't load from disk.")

        if self._directory is None:
            return self._filename
        else:
            return os.path.join(self._directory, self._filename)

    def is_loaded(self) -> bool:
        if self._obj is None:
            return False
        else:
            return True

    def load(self, update: bool = True,
             reload: bool = False) -> bool:
        if self._obj is not None:
            # Object is already loaded
            return True

        # Get full filepath of file
        filepath = self.filepath

        # Load object at filepath
        try:
            self._obj = load_object(filepath, update=update, reload=reload)
            return True
        except Exception as e:
            print("There was an issue loading an object!")
            print(e)
            return False

    @property
    def obj(self):
        if self._obj is None:
            raise RuntimeError(
                "Container has not loaded the object yet")
        return self._obj

    def set_obj(self, obj: DryObject):
        self._obj = obj

    def get_obj(self, load=False):
        if load:
            self.load()
        return self.obj

    def save(self, directory: Optional[str] = None,
             fail_without_directory: bool = True):
        if self._obj is not None:
            # Get object filepath
            filepath = self.filepath
            # Split filepath
            orig_dir, filename = os.path.split(filepath)
            # Correct directory if asked
            if directory is not None:
                new_dir = directory
            else:
                new_dir = orig_dir

            # If we don't want to have no directory, fail here.
            if fail_without_directory and new_dir == '':
                raise RuntimeError(
                    "Instructed not to save an object "
                    "without explicit directory")

            # Build final filepath
            filepath = os.path.join(new_dir, filename)
            self._obj.save_self(filepath)

    def unload(self):
        if self._obj is not None:
            del self._obj
            self._obj = None

    def delete(self):
        # Delete on-disk file
        os.remove(self.filepath)

    def set_directory(self, directory):
        self._directory = directory

    def set_filename(self, filename):
        self._filename = filename

    def definition(self):
        if self._obj is None:
            # We need to load the file from disk
            with DryObjectFile(self.filepath) as f:
                return f.definition()
        else:
            return self._obj.definition()


# This type will act as a fascade for the various DryObject* types.
class DryRepo(object):
    def __init__(self, directory: Optional[str] = None, create: bool = False,
                 load_objects: bool = True, **kwargs):
        super().__init__(**kwargs)

        # A dictionary of objects
        self.obj_dict = {}

        if directory is not None:
            self.link_to_directory(directory, create=create,
                                   load_objects=load_objects)
        else:
            self.directory = None

        self._save_objs_on_deletion = False

    @property
    def save_objs_on_deletion(self):
        return self._save_objs_on_deletion

    @save_objs_on_deletion.setter
    def save_objs_on_deletion(self, val: bool):
        if val and self.directory is None:
            raise RuntimeError(
                "Give the repo a directory if you want it to "
                "save objects upon deletion")
        self._save_objs_on_deletion = val

    def __del__(self):
        if self.save_objs_on_deletion:
            self.save()

    def link_to_directory(self, directory: str, create: bool = False,
                          load_objects: bool = True):
        # Check that directory exists
        if not os.path.exists(directory):
            if create:
                os.makedirs(directory)
        self.directory = directory
        if load_objects:
            self.load_objects_from_directory()

    def __len__(self):
        return len(self.obj_dict)

    def add_obj_cont(self, cont: DryRepoContainer):
        obj_def = cont.definition()
        ind_hash = obj_def.get_individual_id()
        if ind_hash in self.obj_dict:
            raise ValueError(
                f"Object {ind_hash} already exists in the repo!")
        self.obj_dict[ind_hash] = cont

    def load_objects_from_directory(self, directory: Optional[str] = None,
                                    selector: Optional[Callable] = None,
                                    verbose: bool = False):
        "Method to refresh the internal dictionary of objects."
        # Handle directory
        if directory is None:
            if self.directory is None:
                raise RuntimeError(
                    "No default directory selected for this repo!")
            directory = self.directory

        files = os.listdir(directory)

        num_loaded = 0
        for filename in files:
            try:
                # Load container object
                obj_cont = DryRepoContainer.from_filepath(
                    filename, directory=directory
                )
                # Run selector
                if selector is not None:
                    if not selector(obj_cont.definition()):
                        continue
                # Add the object
                self.add_obj_cont(obj_cont)
                num_loaded += 1
            except Exception as e:
                print(f"WARNING! Malformed file found! {obj_cont.filepath} "
                      f"skipping load. Error was: {e}")
        if verbose:
            print(f"Loaded {num_loaded} objects")

    def add_object(self, obj: DryObject, filepath: Optional[str] = None):
        # Add a single object
        if filepath is not None:
            directory, filename = os.path.split(filepath)
            if directory == '':
                if self.directory is not None:
                    directory = self.directory
                else:
                    directory = None
        else:
            filename = None
            directory = self.directory

        obj_cont = DryRepoContainer.from_object(
            obj, directory=directory, filename=filename)
        self.add_obj_cont(obj_cont)

    def add_objects(self, obj_factory: DryObjectFactory, num=1):
        # Create numerous objects from a factory function
        for i in range(num):
            obj = obj_factory()
            self.add_object(obj)

    def make_container_handler(self,
                               load_objects: bool = True,
                               open_container: bool = True,
                               update: bool = True):

        def container_opener(obj_cont, open_container: bool = True):
            if open_container:
                return obj_cont.obj
            else:
                return obj_cont

        def container_handler(obj_cont):
            if not obj_cont.is_loaded():
                if load_objects:
                    obj_cont.load(update=update)

            return container_opener(obj_cont, open_container=open_container)

        return container_handler

    def make_filter_func(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            only_loaded: bool = False):
        if sel_args is None:
            sel_args = []
        if sel_kwargs is None:
            sel_kwargs = {}

        def filter_func(obj_cont):
            if only_loaded:
                if not obj_cont.is_loaded():
                    return False

            if selector is not None:
                if selector(obj_cont.definition(), *sel_args, **sel_kwargs):
                    return True
                else:
                    return False
            else:
                return True

        return filter_func

    def get_obj(
            self,
            obj_def: DryObjectDef):
        ind_id = obj_def.get_individual_id()
        return self.get_obj_by_id(ind_id)

    def get_obj_by_id(
            self,
            ind_id):
        obj_container = self.obj_dict[ind_id]
        return obj_container.obj

    def __contains__(
            self, item: Union[DryObject, DryObjectDef, dict, DryObjectFile]):
        if issubclass(type(item), DryObject) or \
           issubclass(item, DryObjectFile):
            obj_id = item.definition().get_individual_id()
        elif issubclass(type(item), DryObjectDef):
            obj_id = item.get_individual_id()
        elif issubclass(type(item), dict):
            obj_id = DryObjectDef.from_dict(item).get_individual_id()
        else:
            raise TypeError("Unsupported type for repo.contains!")
        return obj_id in self.obj_dict

    def get(self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            load_objects: bool = True,
            only_loaded: bool = False,
            update: bool = True,
            open_container: bool = True,
            verbose: bool = True):
        # Filter the internal object list
        filter_func = \
            self.make_filter_func(
                selector,
                sel_args=sel_args,
                sel_kwargs=sel_kwargs,
                only_loaded=only_loaded)
        obj_list = list(filter(filter_func, self.obj_dict.values()))

        # Build container handler
        container_handler = \
            self.make_container_handler(
                update=update,
                load_objects=load_objects,
                open_container=open_container)
        return list(map(container_handler, obj_list))

    def apply(self,
              func, func_args=None, func_kwargs=None,
              selector: Optional[Callable] = None,
              sel_args=None, sel_kwargs=None,
              verbose: bool = False,
              **kwargs):
        """
        Apply a function to all objects tracked by the repo.
        We can also use a DrySelector to apply only to specific models
        **kwargs is passed to self.get
        """
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = {}

        # Create apply function
        def apply_func(obj):
            return func(obj, *func_args, **func_kwargs)

        # Get object list
        objs = self.get(
            selector=selector,
            sel_args=sel_args, sel_kwargs=sel_kwargs,
            **kwargs)

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

    def reload_objs(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            update: bool = False,
            reload: bool = False,
            as_cls=None):
        if self.directory is None:
            raise RuntimeError("Repo directory needs to be set for reloading.")

        def reload_func(obj_cont):
            if not obj_cont.is_loaded():
                raise RuntimeError("Can only reload already loaded DryObjects")

            # Get the object
            obj = obj_cont.obj

            # Get current definition of class
            if as_cls is not None:
                cls = as_cls
            else:
                cls = type(obj)
            new_cls = get_current_cls(cls, reload=reload)

            # Set object
            obj_cont.set_obj(change_object_cls(obj, new_cls, update=True))

            # Remove old object
            del obj

        self.apply(
            reload_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, only_loaded=True, load_objects=False)

    def save(self,
             selector: Optional[Callable] = None,
             sel_args=None, sel_kwargs=None,
             directory: Optional[str] = None):

        def save_func(obj_cont):
            if not obj_cont.is_loaded():
                raise RuntimeError("Can only save currently loaded DryObject")

            # Save object
            obj_cont.save(directory=directory)

        self.apply(
            save_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, only_loaded=True, load_objects=False)

    def save_and_cache(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None):
        "Save and then delete objects. Replace their entries with strings"

        def save_func(obj_cont):
            if not obj_cont.is_loaded():
                raise RuntimeError("Can only save currently loaded DryObject")

            # Save object
            obj_cont.save()
            obj_cont.unload()

        self.apply(
            save_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, only_loaded=True, load_objects=False)

    def unload(self,
               selector: Optional[Callable] = None,
               sel_args=None, sel_kwargs=None):

        def unload_func(obj_cont):
            if obj_cont.is_loaded():
                obj_cont.unload()

        self.apply(
            unload_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, only_loaded=True, load_objects=False)

    def delete(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            only_loaded: bool = True):
        "Unload and delete from disk selected models"

        # Get all selected objects
        obj_containers = self.get(
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, only_loaded=only_loaded, load_objects=False)

        for obj_cont in obj_containers:
            # Delete object from repo object tracker
            ind_hash = obj_cont.definition().get_individual_id()
            del self.obj_dict[ind_hash]

            # Delete object from disk
            obj_cont.delete()

            # Delete object container
            del obj_cont

    def list_unique_objs(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            only_loaded=False):
        "List unique object definitions yielded by a selector"

        obj_containers = self.get(
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, load_objects=False)

        results = {}

        for obj_cont in obj_containers:
            if only_loaded:
                # Skip unloaded objects
                if not obj_cont.is_loaded():
                    continue
            obj_def = obj_cont.definition()
            obj_cat_def = obj_def.get_cat_def()
            cat_id = obj_cat_def.get_category_id()

            if only_loaded:
                entry = results.get(
                    cat_id,
                    {'def': obj_cat_def, 'num': 0})
                entry['num'] += 1
                results[cat_id] = entry
            else:
                entry = results.get(
                    cat_id,
                    {'def': obj_cat_def, 'loaded': 0, 'unloaded': 0})
                if obj_cont.is_loaded():
                    entry['loaded'] += 1
                else:
                    entry['unloaded'] += 1
                results[cat_id] = entry

        for cat_id in results:
            entry = results[cat_id]
            pprint(entry['def'])
            if only_loaded:
                print(f"{entry['num']} loaded")
            else:
                print(f"{entry['loaded']}/"
                      f"{entry['loaded']+entry['unloaded']} loaded")
