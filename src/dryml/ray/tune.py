import os
import pathlib
import dryml
from dryml.models import TrainSpec
from dryml.models import Trainable
from dryml.data import Dataset
from dryml import Repo
from ray.air import session
from ray.air.checkpoint import Checkpoint
import pickle
import uuid


class TuneObjectSaver(object):
    def __init__(
            self, model: Trainable = None,
            train_state: TrainSpec = None,
            repo: Repo = None,
            test_ds: Dataset = None,
            ctx_reqs=None,
            tmp_checkpoint_dir='/tmp',
            metrics={}):
        if model is None:
            raise ValueError("Must pass a model.")
        self.model = model

        if train_state is None:
            raise ValueError("Must pass a train state.")
        self.train_state = train_state

        if repo is None:
            raise ValueError("Must pass a repo.")
        self.repo = repo

        if test_ds is None:
            raise ValueError("Must pass test data for metrics")
        self.test_ds = test_ds

        if ctx_reqs is None:
            raise ValueError("Must pass ctx_reqs so they can be saved")
        self.ctx_reqs = ctx_reqs

        self.metrics = metrics
        self.tmp_checkpoint_dir = tmp_checkpoint_dir

    def __call__(self):
        # Extract current step from dryml training process
        # step = self.train_state.global_step()

        temp_checkpoint_dir = os.path.join(
            self.tmp_checkpoint_dir,
            str(uuid.uuid4()))
        pathlib.Path(temp_checkpoint_dir).mkdir(exist_ok=False)
        # d = tempfile.TemporaryDirectory()
        # checkpoint_dir = d.name
        # Get checkpoint directory
        # with checkpoint.as_directory() as checkpoint_dir:

        # Create checkpoint dir
        #    if not os.path.exists(checkpoint_dir):
        #       # Create the checkpoint directory if needed
        #       pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Save model to checkpoint
        self.model.save_self(f"{temp_checkpoint_dir}/model.dry")
        self.train_state.save(f"{temp_checkpoint_dir}/train_state.pkl")
        with open(f"{temp_checkpoint_dir}/ctx_reqs.pkl", 'wb') as f:
            f.write(pickle.dumps(self.ctx_reqs))

        checkpoint = Checkpoint.from_directory(path=temp_checkpoint_dir)

        # Compute test set metrics
        metric_values = {}
        for metric_name in self.metrics:
            metric = self.metrics[metric_name]
            value = metric(self.model, self.test_ds)
            metric_values[metric_name] = value
        # Save model's dry_id
        metric_values['dry_id'] = self.model.dry_id

        # Report results
        session.report(metrics=metric_values, checkpoint=checkpoint)


class Trainer(object):
    def __init__(
            self,
            name=None,
            prep_method=None,
            metrics={}):
        self._name = name
        self.prep_method = prep_method
        self.metrics = metrics

    def __call__(self, config, checkpoint_dir=None):
        prep_dict = self.prep_method()
        # Create repo
        repo = prep_dict['repo']()
        model_gen = prep_dict['model']
        data_gen = prep_dict['data']

        # Load or initialize training
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as checkpoint_dir:
                # Load existing train state
                train_state = TrainSpec.load(
                    os.path.join(checkpoint_dir, "train_state.pkl"))

                # Load object from checkpoint
                model = dryml.load_object(
                    os.path.join(checkpoint_dir, "model.dry"),
                    repo=repo)

                # Retrieve requested context requirements
                ctx_filepath = os.path.join(checkpoint_dir, "ctx_reqs.pkl")
                if os.path.exists(ctx_filepath):
                    with open(ctx_filepath, 'rb') as f:
                        ctx_reqs = pickle.loads(f.read())
                else:
                    ctx_reqs = None

        else:
            # Initialize train spec
            train_state = TrainSpec()

            # Create build_obj function
            model_dict = model_gen(config, repo=repo)

            model = model_dict['model']
            if 'ctx_reqs' in model_dict:
                ctx_reqs = model_dict['ctx_reqs']
            else:
                ctx_reqs = model.dry_context_requirements()

        # Get data pipeline reqs
        data_ctx_reqs = prep_dict['data_ctx']()

        # Combine data and model contexts
        ctx_reqs = dryml.context.combine_reqs(
            ctx_reqs, data_ctx_reqs)

        # Acquire context
        dryml.context.set_context(ctx_reqs)

        # Start data pipelines
        data_dict = data_gen()
        train_ds = data_dict['train']
        test_ds = data_dict['test']

        # create train saver callable
        obj_saver = TuneObjectSaver(
            model=model,
            train_state=train_state,
            repo=repo,
            test_ds=test_ds,
            ctx_reqs=ctx_reqs,
            metrics=self.metrics,)
        callbacks = [obj_saver]

        # Prepare model for training
        model.prep_train()

        # Start training
        model.train(
            train_ds,
            train_spec=train_state,
            train_callbacks=callbacks)

        # Save trained model to repo
        repo.add_object(model)
        repo.save()

        # Save metric data
        final_metrics = {}
        for metric_name in self.metrics:
            metric = self.metrics[metric_name]
            metric_val = metric(model, test_ds)
            final_metrics[metric_name] = metric_val
        final_metrics.update(dry_id=model.dry_id, done=True)

        # Report metrics
        session.report(metrics=final_metrics)
