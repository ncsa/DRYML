"""
Code to run subprocesses with proper contexts
"""


from dryml.context.context_tracker import contexts, WrongContextError, \
    context, consolidate_contexts, get_context_requirements, \
    ContextManager
import copy
import multiprocessing as mp
import traceback
import functools
import io
import zipfile
import time
import dill
from dryml.save_cache import SaveCache

mp_ctx = mp.get_context('spawn')


# Wrapping process base starting from
# https://stackoverflow.com/a/33599967/2485932
# This ensures the process will exit even if an error
# is thrown.
class Process(mp_ctx.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = mp_ctx.Pipe()
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


def cls_method_compute(fn_name, **ctx_kwargs):
    """
    Mark a class method as compute. DryMeta will ensure
    such named methods are wrapped in a compute context.
    """

    def _dec(cls):
        cls_method_list = getattr(cls, '__dry_compute_methods__', [])
        cls_method_list.append((fn_name, ctx_kwargs))
        cls.__dry_compute_methods__ = cls_method_list
        return cls

    return _dec


# Get list of dry objects because we need to find a good
# context to use.
def get_dry_objects(*args, **kwargs):
    from dryml import DryObject

    dry_objects = []
    for arg in args:
        if isinstance(arg, DryObject):
            dry_objects.append(arg)
    for name in kwargs:
        arg = kwargs[name]
        if isinstance(arg, DryObject):
            dry_objects.append(arg)
    return dry_objects


# extra methods for activating/deactivating objects
def activate_objects(obj_list):
    for obj in obj_list:
        obj.compute_activate()


class process_executor(object):
    def __init__(
            self,
            f=None,
            ctx_reqs={},
            update_obj_defs=[],
            args=[],
            kwargs={}):
        self.f_ser = dill.dumps(f)
        self.ctx_reqs = ctx_reqs
        self.update_obj_defs = update_obj_defs
        self.args_ser = dill.dumps(args)
        self.kwargs_ser = dill.dumps(kwargs)

    def final_call(
            self,
            f,
            ctx_send_q,
            ctx_ret_q,
            *args,
            **kwargs):

        # import resource
        # max_mem = 4*1024*1024*1024
        # resource.setrlimit(
        #     resource.RLIMIT_AS,
        #     (max_mem, resource.RLIM_INFINITY))

        if ctx_send_q is None:
            raise RuntimeError("A send queue is required.")

        if ctx_ret_q is None:
            raise RuntimeError("A return queue is required.")

        # Activate context
        with ContextManager(resource_requests=self.ctx_reqs) as ctx_mgr:
            # Get list of dry_objects
            dry_objects = get_dry_objects(*args, **kwargs)

            # Activate unactivated objects
            activate_objects(dry_objects)

            # Execute method
            res = f(*args, **kwargs)

            # Put result in return queue
            ctx_ret_q.put(res)

            save_cache = SaveCache()

            # Put object updates in queue
            if len(self.update_obj_defs) > 0:
                for obj_def in self.update_obj_defs:
                    found = False
                    for obj in dry_objects:
                        if obj_def == obj.definition():
                            res_buf = io.BytesIO()
                            obj.save_self(res_buf, save_cache=save_cache)
                            res_buf.seek(0)
                            ctx_ret_q.put(res_buf.read())
                            found = True
                            break
                    # Close int_files in save_cache
                    if not found:
                        raise RuntimeError(
                            "Couldn't find an object to associate"
                            " a definition to!")

            ctx_mgr.deactivate_objects(save_cache=save_cache)

            # Wait until return queue is empty (parent thread has emptied it)
            while not ctx_ret_q.empty():
                # Sleep a bit
                time.sleep(0.1)

        # Delete save cache
        del save_cache

    def __call__(self, ctx_send_q, ctx_ret_q):
        # Undill function
        f = dill.loads(self.f_ser)

        # Undill args/kwargs
        args = dill.loads(self.args_ser)
        kwargs = dill.loads(self.kwargs_ser)

        ph_data = ctx_send_q.get()

        from dryml.dry_object import reconstruct_args_kwargs
        reconstruct_args_kwargs(args, kwargs, ph_data)

        self.final_call(f, ctx_send_q, ctx_ret_q, *args, **kwargs)


def compute_context(
        ctx_context_reqs=None,
        ctx_use_existing_context=True,
        ctx_dont_create_context=False,
        ctx_update_objs=False,
        ctx_verbose=False):

    """
        ctx_context_reqs: if None, will try to guess
        ctx_use_existing_context: if True, will try to run in
            the currently active context without a subprocess
        ctx_dont_create_context: if True, will avoid trying to create
            a context, instead relying on the existing context.
        ctx_update_objs: if True, will serialize objects after method is run
            and update local objects by running load_object with
            the serialized results
    """

    def _func_dec(f):
        if hasattr(f, '__dry_context_wrapped__'):
            # Don't wrap a function twice
            if f.__dry_context_wrapped__:
                return f

        @functools.wraps(f)
        def wrapped_func(
                *args,
                call_context_reqs=None,
                call_use_existing_context=None,
                call_dont_create_context=None,
                call_update_objs=None,
                call_update_skiplist=None,
                call_verbose=None,
                **kwargs):
            """
                call_*: call version of the ctx variables which take
                    precedence.
                call_update_skiplist: A list of objects to skip
                    when updating objects.
            """

            # Check whether we are using an existing context
            use_existing_context = ctx_use_existing_context
            if call_use_existing_context is not None:
                use_existing_context = call_use_existing_context
            dont_create_context = ctx_dont_create_context
            if call_dont_create_context is not None:
                dont_create_context = call_dont_create_context

            verbose = ctx_verbose
            if call_verbose is not None:
                verbose = call_verbose

            ctx_reqs = ctx_context_reqs
            if call_context_reqs is not None:
                ctx_reqs = call_context_reqs

            if ctx_reqs is None:
                # Determine needed contexts
                ctx_reqs = get_context_requirements(get_dry_objects(
                    *args, **kwargs))

            if use_existing_context:
                ctx_manager = context()
                if ctx_manager is not None:
                    if not ctx_manager.satisfies(ctx_reqs):
                        raise WrongContextError(
                            "Current context is not appropriate")
                else:
                    # No currently active context, need to create one
                    use_existing_context = False

            if use_existing_context:
                # Execute the method in this thread.
                # Activate objects which don't have a context active.
                activate_objects(get_dry_objects(
                    *args, **kwargs))

                # Execute method
                res = f(*args, **kwargs)

                # We don't have to wory about any of the
                # context management stuff

                return res
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

                ctx_ret_q = mp_ctx.Queue()
                ctx_send_q = mp_ctx.Queue()

                from dryml.dry_object import prep_args_kwargs

                # Replace DryObjects in args/kwargs with placeholders
                # Get placeholder data
                (args, kwargs), ph_data = prep_args_kwargs(args, kwargs)

                executor = process_executor(
                    f=f, ctx_reqs=ctx_reqs,
                    update_obj_defs=update_obj_defs,
                    args=args,
                    kwargs=kwargs)

                # run function
                p = Process(target=executor, args=[ctx_send_q, ctx_ret_q])
                p.start()

                # Send placeholder data
                ctx_send_q.put(ph_data)

                if verbose:
                    print(f"Started context isolation process, pid: {p.pid}")

                # Check loop
                queue_results = []

                # Num elements which need to be fetched before joining
                num_to_fetch = 1
                num_to_fetch += len(update_objs_list)

                def check_queue():
                    while len(queue_results) < num_to_fetch:
                        # Check whether the queue is empty
                        if not ctx_ret_q.empty():
                            # Get result
                            queue_results.append(ctx_ret_q.get())
                        else:
                            break

                while p.is_alive():
                    # check for exception
                    if p.exception is not None:
                        e, tb = p.exception
                        print(
                            "Exception encountered in context "
                            f"thread! pid: {p.pid}")
                        print(tb)
                        # rejoin thread
                        p.join()
                        # Reraise
                        raise e

                    # Get results, we need to do this before the join
                    # to prevent deadlock from large data being shuttled
                    # through the queue.
                    check_queue()

                    # Sleep a bit
                    time.sleep(0.1)

                if p.exception is not None:
                    e, tb = p.exception
                    print("Exception encountered in context "
                          f"thread! pid: {p.pid}")
                    print(tb)
                    # rejoin thread
                    p.join()
                    # Reraise
                    raise e

                # Check queue again
                check_queue()

                # join the thread
                p.join()

                if len(queue_results) < num_to_fetch:
                    raise RuntimeError("Didn't get all expected results!")

                # retrieve return value
                retval = queue_results.pop(0)

                # Update dry objects
                load_issue_list = []
                for obj in update_objs_list:
                    obj_buf = queue_results.pop(0)
                    obj_buf = io.BytesIO(obj_buf)
                    from dryml.dry_object import load_object_content
                    if not load_object_content(obj, obj_buf):
                        load_issue_list.append(obj)

                # Close thread
                p.close()

                # Close and Delete the temporary queue
                ctx_ret_q.close()
                del ctx_ret_q

                if len(load_issue_list):
                    print("There was an issue updating the following objects")
                    for obj in load_issue_list:
                        print(obj)

                if verbose:
                    print(f"Ended context isolation process, pid: {p.pid}")

                return retval

        wrapped_func.__dry_context_wrapped__ = True

        # Return wrapped function
        return wrapped_func

    return _func_dec


def compute(f):
    return compute_context()(f)


class tune_process_executor(object):
    def __init__(
            self,
            f=None,
            ctx_name=None,
            context_kwargs={},
            update_obj_defs=[],
            args=[],
            kwargs={}):
        self.f_ser = dill.dumps(f)
        self.ctx_name = ctx_name
        self.context_kwargs = context_kwargs
        self.update_obj_defs = update_obj_defs
        self.args_ser = dill.dumps(args)
        self.kwargs_ser = dill.dumps(kwargs)
        self._dry_objects = None
        self._activated_objects = None

    def final_call(
            self,
            f,
            ctx_ret_q,
            tune_report_q,
            checkpoint_req_q,
            checkpoint_ret_q,
            *args,
            **kwargs):

        if ctx_ret_q is None:
            raise RuntimeError("A return queue is required.")

        if tune_report_q is None:
            raise RuntimeError("A report queue is required.")

        if checkpoint_req_q is None:
            raise RuntimeError("A checkpoint request queue is required.")

        if checkpoint_ret_q is None:
            raise RuntimeError("A checkpoint return queue is required.")

        def cleanup_process(ret_val):
            # Put result in return queue
            ctx_ret_q.put(ret_val)

            # Put object updates in queue
            if len(self.update_obj_defs) > 0:
                for obj_def in self.update_obj_defs:
                    found = False
                    for obj in self._dry_objects:
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

            # Wait until return queue is empty
            # (parent thread has emptied it)
            while not ctx_ret_q.empty():
                # Sleep a bit
                time.sleep(0.1)

            # Wait until tune report queue is empty
            # (parent thread has emptied it)
            while not tune_report_q.empty():
                time.sleep(0.1)

            # Wait until checkpoint request queue is empty
            # (parent thread should empty it.)
            while not checkpoint_req_q.empty():
                time.sleep(0.1)

        # Activate context
        with contexts[self.ctx_name][1](**self.context_kwargs):
            # Get list of dry_objects
            self._dry_objects = get_dry_objects(*args, **kwargs)
            # Activate unactivated objects
            self._activated_objects = activate_objects(self._dry_objects)

            try:
                res = f(
                    *args, **kwargs, tune_report_q=tune_report_q,
                    checkpoint_req_q=checkpoint_req_q,
                    checkpoint_ret_q=checkpoint_ret_q)
                cleanup_process(res)
            except KeyboardInterrupt:
                cleanup_process(None)

    def __call__(
            self, ctx_ret_q, tune_report_q=None,
            checkpoint_ret_q=None, checkpoint_req_q=None):
        # Undill function
        f = dill.loads(self.f_ser)

        # Undill args/kwargs
        args = dill.loads(self.args_ser)
        kwargs = dill.loads(self.kwargs_ser)

        self.final_call(
            f, ctx_ret_q, tune_report_q, checkpoint_req_q,
            checkpoint_ret_q, *args, **kwargs)


def tune_compute_context(
        ctx_context_type=None,
        ctx_use_existing_context=True,
        ctx_dont_create_context=False,
        ctx_update_objs=False,
        ctx_verbose=False,
        **ctx_context_kwargs):

    """
        A compute context allowing the retrieval of intermediate results
        to be sent to tune during training.
        ctx_context_type: if None, will try to guess
        ctx_use_existing_context: if True, will try to run in
            the currently active context without a subprocess
        ctx_dont_create_context: if True, will avoid trying to create
            a context, instead relying on the existing context.
        ctx_update_objs: if True, will serialize objects after method is run
            and update local objects by running load_object with
            the serialized results
    """

    def _func_dec(f):
        if hasattr(f, '__dry_context_wrapped__'):
            # Don't wrap a function twice
            if f.__dry_context_wrapped__:
                return f

        nonlocal ctx_context_kwargs
        if ctx_context_kwargs is None:
            ctx_context_kwargs = {}

        @functools.wraps(f)
        def wrapped_func(
                *args,
                call_context_type=None,
                call_update_objs=None,
                call_update_skiplist=None,
                call_context_kwargs=None,
                call_verbose=None,
                **kwargs):
            """
                call_*: call version of the ctx variables which take
                    precedence.
                call_update_skiplist: A list of objects to skip
                    when updating objects.
            """
            from ray import tune

            verbose = ctx_verbose
            if call_verbose is not None:
                verbose = call_verbose

            ctx_name = ctx_context_type
            if call_context_type is not None:
                ctx_name = call_context_type

            if ctx_name is None:
                # Determine context type
                ctxs = []
                for obj in get_dry_objects(*args, **kwargs):
                    ctxs.append(obj.dry_compute_context())
                ctx_name = consolidate_contexts(ctxs)

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

            # Create return queue
            ctx_ret_q = mp_ctx.Queue()

            executor = tune_process_executor(
                f=f, ctx_name=ctx_name,
                context_kwargs=context_kwargs,
                update_obj_defs=update_obj_defs,
                args=args,
                kwargs=kwargs)

            # Create tune report queue
            tune_report_q = mp_ctx.Queue()

            # Create checkpoint queues
            checkpoint_req_q = mp_ctx.Queue()
            checkpoint_ret_q = mp_ctx.Queue()

            # run function
            p = Process(
                target=executor, args=[ctx_ret_q],
                kwargs={
                    'tune_report_q': tune_report_q,
                    'checkpoint_req_q': checkpoint_req_q,
                    'checkpoint_ret_q': checkpoint_ret_q})
            p.start()

            if verbose:
                print(f"Started context isolation process, pid: {p.pid}")

            # Check loop
            queue_results = []

            # Num elements which need to be fetched before joining
            num_to_fetch = 1
            num_to_fetch += len(update_objs_list)

            def check_queues():
                nonlocal tune_report_q
                nonlocal ctx_ret_q
                nonlocal checkpoint_req_q
                nonlocal checkpoint_ret_q
                while len(queue_results) < num_to_fetch or \
                        not tune_report_q.empty():
                    # Check whether the queue is empty
                    if not ctx_ret_q.empty():
                        # Get result
                        queue_results.append(ctx_ret_q.get())
                    elif not tune_report_q.empty():
                        report_dict = tune_report_q.get()
                        if type(report_dict) is not dict:
                            raise ValueError(
                                "Value in tune report queue was "
                                "not a dictionary!")
                        # Report result to tune
                        tune.report(**report_dict)
                    elif not checkpoint_req_q.empty():
                        step_req = checkpoint_req_q.get()
                        with tune.checkpoint_dir(step=step_req) as d:
                            checkpoint_ret_q.put(d)
                    else:
                        break

            while p.is_alive():
                # check for exception
                if p.exception is not None:
                    e, tb = p.exception
                    print(
                        "Exception encountered in context "
                        f"thread! pid: {p.pid}")
                    print(tb)
                    # rejoin thread
                    p.join()
                    # Reraise
                    raise e

                # Get results, we need to do this before the join
                # to prevent deadlock from large data being shuttled
                # through the queue.
                check_queues()

                # Sleep a bit
                time.sleep(0.1)

            if p.exception is not None:
                e, tb = p.exception
                print("Exception encountered in context "
                      f"thread! pid: {p.pid}")
                print(tb)
                # rejoin thread
                p.join()
                # Reraise
                raise e

            # Check queue again
            check_queues()

            # join the thread
            p.join()

            if len(queue_results) < num_to_fetch:
                raise RuntimeError("Didn't get all expected results!")

            # retrieve return value
            retval = queue_results.pop(0)

            # Update dry objects
            for obj in update_objs_list:
                obj_buf = queue_results.pop(0)
                zf = zipfile.ZipFile(io.BytesIO(obj_buf), mode='r')
                obj.load_object(zf)

            # Close thread
            p.close()

            # Close and Delete the temporary queue
            ctx_ret_q.close()
            del ctx_ret_q

            # Close and Delete the tune report queue
            tune_report_q.close()
            del tune_report_q

            # Close and Delete the checkpoint queues
            checkpoint_req_q.close()
            del checkpoint_req_q

            # Wait until the checkpoint return queue is empty
            while not checkpoint_ret_q.empty():
                time.sleep(0.1)

            checkpoint_ret_q.close()
            del checkpoint_ret_q

            if verbose:
                print(f"Ended context isolation process, pid: {p.pid}")

            return retval

        wrapped_func.__dry_context_wrapped__ = True

        # Return wrapped function
        return wrapped_func

    return _func_dec
