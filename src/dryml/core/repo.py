import os
import traceback
from dryml.core.object import Object, ObjectFactory, ObjectFile, \
    ObjectDef, change_object_cls, load_object, get_contained_objects
from dryml.core.config import MissingIdError
from dryml.core.selector import Selector
from dryml.core.utils import get_current_cls
from typing import Optional, Callable, Union, Mapping
import tqdm
from pprint import pprint


RepoKey = Union[Object, ObjectDef, dict, ObjectFile, Selector, str]


class RepoContainer(object):
    @staticmethod
    def from_filepath(filename: str, directory: Optional[str] = None):
        new_container = RepoContainer(directory=directory)
        new_container._filename = filename
        return new_container

    @staticmethod
    def from_object(obj: Object, filename: Optional[str] = None,
                    directory: Optional[str] = None):
        new_container = RepoContainer(directory=directory)
        new_container._obj = obj
        if filename is not None:
            new_container._filename = filename
        else:
            # Set default filename
            new_container._filename = \
                obj.dry_id+'.dry'
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

        return f"RepoContainer: {obj_part} {def_part} at {path_part}"

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

    def set_obj(self, obj: Object):
        self._obj = obj

    def get_obj(self, load=False):
        if load:
            if not self.load():
                raise RuntimeError("There was an issue loading the object!")
        return self.obj

    def save(self, directory: Optional[str] = None,
             fail_without_directory: bool = True,
             save_cache=None):
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
        # Delete on-disk file if it exists
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def set_directory(self, directory):
        self._directory = directory

    def set_filename(self, filename):
        self._filename = filename

    def get_contained_objects(self):
        if self._obj is None:
            return set()

        return get_contained_objects(self._obj)

    def definition(self):
        if self._obj is None:
            # We need to load the file from disk
            with ObjectFile(self.filepath) as f:
                return f.definition()
        else:
            return self._obj.definition()


# This type will act as a fascade for the various Object* types.
class Repo(object):
    def __init__(self, directory: Optional[str] = None, create: bool = False,
                 load_objects: bool = True, **kwargs):
        super().__init__(**kwargs)

        # A dictionary of objects
        self.obj_dict = {}

        self._save_objs_on_deletion = False

        if directory is not None:
            self.link_to_directory(directory, create=create,
                                   load_objects=load_objects)
        else:
            self.directory = None

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

    def add_obj_cont(self, cont: RepoContainer):
        obj_id = cont.definition().dry_id
        if obj_id in self.obj_dict:
            raise ValueError(
                f"Object {obj_id} already exists in the repo!")
        self.obj_dict[obj_id] = cont

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
                obj_cont = RepoContainer.from_filepath(
                    filename, directory=directory
                )
                # Run selector
                if selector is not None:
                    if not selector(obj_cont.definition()):
                        continue
                if obj_cont.definition().dry_id not in self.obj_dict:
                    # Add the object
                    self.add_obj_cont(obj_cont)
                    num_loaded += 1
            except Exception as e:
                print(f"WARNING! Malformed file found! {obj_cont.filepath} "
                      f"skipping load. Error was: {e}")
                if verbose:
                    print(traceback.format_exc())
        if verbose:
            print(f"Loaded {num_loaded} objects")

    def add_object(self, obj: Object, filepath: Optional[str] = None,
                   add_nested: bool = True):
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

        # Add other nested objects into the repository too
        if add_nested:
            if hasattr(obj, '__dry_obj_container_list__'):
                for o in obj.__dry_obj_container_list__:
                    if o not in self:
                        self.add_object(o, add_nested=add_nested)

        obj_cont = RepoContainer.from_object(
            obj, directory=directory, filename=filename)
        self.add_obj_cont(obj_cont)

    def add_objects(self, obj_factory: ObjectFactory, num=1):
        # Create numerous objects from a factory function
        for i in range(num):
            obj = obj_factory(repo=self)
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
                    if not obj_cont.load(update=update):
                        raise RuntimeError(
                            f"Could not load object {obj_cont.filepath}!")

            return container_opener(obj_cont, open_container=open_container)

        return container_handler

    def make_filter_func(
            self,
            selector: Optional[Union[Callable, ObjectDef,
                                     Mapping, Object]] = None,
            sel_args=None, sel_kwargs=None,
            only_loaded: bool = False):
        if sel_args is None:
            sel_args = []
        if sel_kwargs is None:
            sel_kwargs = {}

        if type(selector) is ObjectDef:
            # Wrap automatically for convenience
            selector = Selector.build(selector)

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
            obj_def: ObjectDef,
            load=False):
        ind_id = obj_def.dry_id
        return self.get_obj_by_id(ind_id, load=load)

    def get_obj_by_id(
            self,
            obj_id,
            load=True):
        obj_container = self.obj_dict[obj_id]
        if load:
            if not obj_container.load():
                raise RuntimeError("There was an issue loading the object!")
        return obj_container.obj

    def __contains__(
            self, item: Union[Object, ObjectDef, dict, ObjectFile]):
        if issubclass(type(item), Object) or \
           type(item) is ObjectFile:
            obj_id = item.definition().dry_id
        elif issubclass(type(item), ObjectDef):
            obj_id = item.dry_id
        elif issubclass(type(item), dict):
            obj_id = ObjectDef.from_dict(item).dry_id
        else:
            raise TypeError(
                f"Unsupported type {type(item)} for repo.contains!")
        return obj_id in self.obj_dict

    def __getitem__(
            self, key: Union[RepoKey, list, tuple, Callable]):
        """
        Easy access to objects within.

        if unpack is true, plain objects are returned
        """
        return self.get(key, open_container=True)

    def get(self,
            selector: Optional[Union[RepoKey, list, tuple, Callable]] = None,
            sel_args=None, sel_kwargs=None,
            load_objects: bool = True,
            only_loaded: bool = False,
            update: bool = True,
            open_container: bool = True,
            verbose: bool = True,
            build_missing_def=False):

        # First, handle all cases where the selector refers to a specific
        # object
        obj_id = None
        if selector is not None and \
                (isinstance(selector, Object) or
                 isinstance(selector, ObjectFile)):
            obj_id = selector.definition().dry_id
        elif selector is not None and \
                isinstance(selector, ObjectDef):
            try:
                obj_id = selector.dry_id
            except MissingIdError:
                pass
        elif selector is not None and \
                isinstance(selector, Selector):
            pass
        elif selector is not None and \
                issubclass(type(selector), dict):
            try:
                obj_id = ObjectDef.from_dict(selector).dry_id
            except MissingIdError:
                pass
        elif selector is not None and \
                type(selector) is str:
            obj_id = selector

        # Build container handler
        container_handler = \
            self.make_container_handler(
                update=update,
                load_objects=load_objects,
                open_container=open_container)

        def def_builder(sel):
            new_obj = selector.build(repo=self)
            self.add_object(new_obj)
            return new_obj

        if obj_id is not None:
            # We have a single object request
            if obj_id not in self.obj_dict:
                if type(selector) is ObjectDef and build_missing_def:
                    return def_builder(selector)
                raise KeyError(
                    f"Object {selector} (type: {type(selector)}) "
                    f"(dry_id: {obj_id}) not in the Repository.")

            obj_cont = self.obj_dict[obj_id]
            if only_loaded:
                if not obj_cont.is_loaded():
                    # No objects were found
                    return []
            result = container_handler(obj_cont)
            return result

        # Now handle all cases where we have collections of keys to query

        if type(selector) is list or \
           type(selector) is tuple:

            # We need to combine objects from each key.
            result_set = set()
            for sub_key in selector:
                try:
                    res = self.get(
                        sub_key,
                        sel_args=sel_args,
                        sel_kwargs=sel_kwargs,
                        load_objects=load_objects,
                        only_loaded=only_loaded,
                        update=update,
                        open_container=open_container,
                        verbose=verbose)
                except KeyError:
                    # Skip element
                    continue

                if res is None:
                    # Skip it if None
                    continue
                if type(res) is list:
                    # Multiple results from this key
                    for el in res:
                        result_set.add(el)
                else:
                    # A single key
                    result_set.add(el)

            results = list(result_set)
            if len(results) == 0:
                # handle the case of an empty list
                raise KeyError(
                    f"Key Collection {selector} didn't match any object!")
            elif len(results) == 1:
                # Handle a single result
                return results[0]
            return list(result_set)

        # Now we do the 'vector' methods

        # TODO: consider adding code to set verbosity of the Selector
        # Filter the internal object list
        filter_func = \
            self.make_filter_func(
                selector,
                sel_args=sel_args,
                sel_kwargs=sel_kwargs,
                only_loaded=only_loaded)
        obj_list = list(filter(filter_func, self.obj_dict.values()))

        results = list(map(container_handler, obj_list))
        if len(results) == 0:
            if isinstance(selector, ObjectDef) and build_missing_def:
                return def_builder(selector)
            raise KeyError(f"Key {selector} didn't match any object!")
        elif len(results) == 1:
            return results[0]
        else:
            return results

    def apply(self,
              func, func_args=None, func_kwargs=None,
              selector: Optional[Callable] = None,
              sel_args=None, sel_kwargs=None,
              verbose: bool = False,
              **kwargs):
        """
        Apply a function to all objects tracked by the repo.
        We can also use a Selector to apply only to specific models
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

        if type(objs) is list:
            # apply function to objects
            if verbose:
                return list(map(apply_func, tqdm(objs)))
            else:
                return list(map(apply_func, objs))
        else:
            return apply_func(objs)

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
                raise RuntimeError("Can only reload already loaded Objects")

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
             selector: Optional[Union[RepoKey, list, tuple, Callable]] = None,
             sel_args=None, sel_kwargs=None,
             directory: Optional[str] = None,
             recursive=True,
             error_on_none=False):

        """
        Saves the object or objects matching the input selector to disk.
        if a single object is passed and not already in the repo, it's added
        then saved.

        error_on_none: Whether to throw the KeyError, when no object is in
            the repo matching the key.
        """

        save_cache = set()

        def save_func(obj_or_cont):
            if type(obj_or_cont) is Object:
                # we have a plain dry object

                if obj_or_cont in save_cache:
                    # don't need to save, it's already done.
                    return

                if recursive:
                    contained_objs = get_contained_objects(obj_or_cont)

                    for obj in contained_objs:
                        if obj in self:
                            sub_obj_cont = self.get(obj, open_container=False)
                            save_func(sub_obj_cont)
                        else:
                            save_func(obj)

                # Save
                save_path = os.path.join(
                    directory, f"{obj_or_cont.dry_id}.dry")
                obj_or_cont.save_self(save_path)

                save_cache.add(obj_or_cont)

            else:
                # We have an object container.
                if not obj_or_cont.is_loaded():
                    raise RuntimeError(
                        "Can only save currently loaded Object")

                if obj_or_cont.obj in save_cache:
                    # don't need to save, it's already done.
                    return

                # Get/save contained objects
                if recursive:
                    contained_objs = obj_or_cont.get_contained_objects()

                    for obj in contained_objs:
                        if obj in self:
                            sub_obj_cont = self.get(obj, open_container=False)
                            save_func(sub_obj_cont)
                        else:
                            obj.save_self(os.path.join(
                                directory, f"{obj.dry_id}.dry"))

                # Save object
                obj_or_cont.save(directory=directory, save_cache=save_cache)
                save_cache.add(obj_or_cont.obj)

        # If we haven't added the object to the repo yet, add it now.
        if issubclass(type(selector), Object):
            if selector not in self:
                self.add_object(selector)

        try:
            self.apply(
                save_func,
                selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
                open_container=False, only_loaded=True, load_objects=False)
        except KeyError as e:
            if error_on_none:
                raise e

    def save_by_id(self,
                   obj_id, directory: Optional[str] = None):
        if directory is None:
            directory = self.directory
        self.obj_dict[obj_id].save(directory=directory)

    def save_and_cache(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None):
        "Save and then delete objects. Replace their entries with strings"

        def save_func(obj_cont):
            if not obj_cont.is_loaded():
                raise RuntimeError("Can only save currently loaded Object")

            # Save object
            obj_cont.save()
            obj_cont.unload()

        self.apply(
            save_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, only_loaded=True, load_objects=False)

    def update(self,
               obj: Object):
        """
        Update existing object entry with new object.
        if object doesn't exist, create entry for it.
        """

        obj_id = obj.definition().dry_id

        if obj_id not in self.obj_dict:
            # Add the object as it doesn't exist yet.
            self.add_object(obj)
        else:
            # Set the object for existing container.
            self.obj_dict[obj_id].set_obj(obj)

        # Save object to disk.
        self.save_by_id(obj_id)

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

        def del_cont(obj_cont):
            # Delete object from repo object tracker
            obj_id = obj_cont.definition().dry_id
            del self.obj_dict[obj_id]

            # Delete object from disk
            obj_cont.delete()

            # Delete object container
            del obj_cont

        if type(obj_containers) is list:
            for obj_cont in obj_containers:
                del_cont(obj_cont)
        else:
            del_cont(obj_containers)

    def list_unique_objs(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            only_loaded=False):
        "List unique object definitions yielded by a selector"
        obj_containers = self.get(
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, load_objects=False)

        from dryml.utils import count
        if count(obj_containers) == 1:
            obj_containers = [obj_containers]

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
