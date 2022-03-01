"""
Code to run subprocesses with proper contexts
"""


from dryml.context.context_tracker import get_context_class, \
    contexts, WrongContextError, context, consolidate_contexts
import copy
import multiprocessing as mp
import traceback
import inspect
import functools
import io
import zipfile


# Wrapping process base starting from
# https://stackoverflow.com/a/33599967/2485932
# This ensures the process will exit even if an error
# is thrown.
class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            super().run()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def compute_context(
        ctx_context_type=None,
        ctx_use_existing_context=True,
        ctx_dont_create_context=False,
        ctx_update_objs=False,
        ctx_has_retval=None,
        **ctx_context_kwargs):

    """
        ctx_context_type: if None, will try to guess
        ctx_use_existing_context: if True, will try to run in
            the currently active context without a subprocess
        ctx_dont_create_context: if True, will avoid trying to create
            a context, instead relying on the existing context.
        ctx_update_objs: if True, will serialize objects after method is run
            and update local objects by running load_object with
            the serialized results
        ctx_has_retval: indicates whether there's a return value to get.
            if None, will determine with function signature
    """

    def _func_dec(f):
        # Get function signature for later use
        f_sig = inspect.signature(f)

        # Determine whether function has a return value
        has_retval = ctx_has_retval
        if ctx_has_retval is None:
            if f_sig.return_annotation is not inspect.Signature.empty:
                has_retval = True
            else:
                has_retval = False

        nonlocal ctx_context_kwargs
        if ctx_context_kwargs is None:
            ctx_context_kwargs = {}

        @functools.wraps(f)
        def wrapped_func(
                *args,
                call_context_type=None,
                call_use_existing_context=None,
                call_dont_create_context=None,
                call_update_objs=None,
                call_update_skiplist=None,
                call_context_kwargs=None,
                **kwargs):
            """
                call_*: call version of the ctx variables which take
                    precedence.
                call_update_skiplist: A list of objects to skip
                    when updating objects.
            """

            # Make sure this function knows what DryObject is.
            from dryml import DryObject

            # Check whether we are using an existing context
            use_existing_context = ctx_use_existing_context
            if call_use_existing_context is not None:
                use_existing_context = call_use_existing_context

            # Combine dont create context
            dont_create_context = ctx_dont_create_context
            if call_dont_create_context is not None:
                dont_create_context = call_dont_create_context

            ctx_name = ctx_context_type
            if call_context_type is not None:
                ctx_name = call_context_type

            # Get list of dry objects because we need to find a good
            # context to use.
            def get_dry_objects(*args, **kwargs):
                dry_objects = []
                for arg in args:
                    if isinstance(arg, DryObject):
                        dry_objects.append(arg)
                for name in kwargs:
                    arg = kwargs[name]
                    if isinstance(arg, DryObject):
                        dry_objects.append(arg)
                return dry_objects

            if ctx_name is None:
                # Determine context type
                ctxs = []
                for obj in get_dry_objects(*args, **kwargs):
                    ctxs.append(obj.dry_compute_context())
                ctx_name = consolidate_contexts(ctxs)

            if use_existing_context:
                cur_ctx = context()
                target_ctx_cls = get_context_class(ctx_name)
                if cur_ctx is not None:
                    cur_ctx_type = type(cur_ctx)
                    if not issubclass(cur_ctx_type, target_ctx_cls):
                        raise WrongContextError(
                            "Current context not appropriate")
                else:
                    # No currently active context, need to create one
                    use_existing_context = False

            # extra methods for activating/deactivating objects
            def activate_objects(obj_list):
                activated_objects = []
                for obj in obj_list:
                    if not obj.__dry_compute_mode__:
                        obj.compute_prepare()
                        obj.load_compute()

                return activated_objects

            def deactivate_objects(obj_list):
                for obj in obj_list:
                    obj.save_compute()
                    obj.compute_cleanup()

            if use_existing_context:
                # Execute the method in this thread.
                # Activate objects which don't have a context active.
                activated_objects = activate_objects(get_dry_objects(
                    *args, **kwargs))

                # Execute method
                res = f(*args, **kwargs)

                # Deactivate those objects again.
                deactivate_objects(activated_objects)

                # We don't have to wory about any of the
                # context management stuff

                # return
                if has_retval:
                    return res
                else:
                    return
            else:
                if dont_create_context:
                    raise RuntimeError("Instructed to not create a context!")

                # Execute the method in another thread.
                # get a list of objects we will need to update.
                update_objs = False
                if call_update_objs is not None:
                    update_objs = call_update_objs
                elif ctx_update_objs:
                    update_objs = True

                # record which objects to update
                update_objs_list = []
                update_obj_defs = []
                if update_objs:
                    # build update skiplist
                    if call_update_skiplist is None:
                        update_skiplist = []
                    else:
                        update_skiplist = call_update_skiplist

                    dry_objects = get_dry_objects(*args, **kwargs)

                    # Get list of dry objects
                    update_objs_list = list(filter(
                        lambda o: o not in update_skiplist,
                        dry_objects))

                    # Translate direct list into definitions
                    update_obj_defs = list(map(
                        lambda o: o.definition(),
                        update_objs_list))

                # Prepare context args
                if call_context_kwargs is None:
                    call_context_kwargs = {}

                context_kwargs = copy.copy(ctx_context_kwargs)
                context_kwargs.update(call_context_kwargs)

                if has_retval or len(update_obj_defs) > 0:
                    ctx_ret_q = mp.Queue()

                # Define function for process
                def process_func(
                        *args,
                        ctx_ret_q=None,
                        **kwargs):
                    # Activate context
                    with contexts[ctx_name][1](**context_kwargs):
                        # Get list of dry_objects
                        dry_objects = get_dry_objects(*args, **kwargs)
                        # Activate unactivated objects
                        activated_objects = activate_objects(dry_objects)

                        # Execute method
                        res = f(*args, **kwargs)

                        # Deactivate activated objects
                        deactivate_objects(activated_objects)

                        # Put result in return queue
                        if has_retval:
                            ctx_ret_q.put(res)

                        # Put object updates in queue
                        if len(update_obj_defs) > 0:
                            for obj_def in update_obj_defs:
                                found = False
                                for obj in dry_objects:
                                    if obj_def == obj.definition():
                                        res_buf = io.BytesIO()
                                        obj.save_self(res_buf)
                                        res_buf.seek(0)
                                        ctx_ret_q.put(res_buf.read())
                                        found = True
                                        break
                                if not found:
                                    raise RuntimeError(
                                        "Couldn't find an object to associate"
                                        " a definition to!")

                # If necessary add the queue
                call_kwargs = copy.copy(kwargs)
                if has_retval or len(update_obj_defs) > 0:
                    call_kwargs['ctx_ret_q'] = ctx_ret_q

                # run function
                p = Process(target=process_func, args=args, kwargs=call_kwargs)
                p.start()
                p.join()

                # Exception handling
                if p.exception is not None:
                    e, tb = p.exception
                    print("Exception encountered in context thread!")
                    print(tb)
                    # Reraise
                    raise e

                # retrieve return value
                if has_retval:
                    retval = ctx_ret_q.get()

                # Update dry objects
                for obj in update_objs_list:
                    obj_buf = ctx_ret_q.get()
                    zf = zipfile.ZipFile(io.BytesIO(obj_buf), mode='r')
                    obj.load_object(zf)

                # Delete the temporary queue
                if has_retval or len(update_objs_list) > 0:
                    del ctx_ret_q

                # Return final value
                if has_retval:
                    return retval
                else:
                    return

        # Return wrapped function
        return wrapped_func

    return _func_dec


def compute(f):
    return compute_context()(f)
