from dryml.dry_config import DryMeta
from dryml.models.dry_component import DryComponent
import xgboost
import tempfile
import zipfile


class Model(DryComponent):
    @DryMeta.collect_kwargs
    def __init__(self, **kwargs):
        # It is subclass's responsibility to fill this
        # attribute with an actual keras class
        self.mdl = None
        self.mdl_kwargs = kwargs

    def compute_prepare_imp(self):
        self.mdl = self.cls(**self.mdl_kwargs)

    def compute_cleanup_imp(self):
        self.mdl = None

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        save_file_name = 'model.text'
        # Load Model
        if save_file_name not in file.namelist():
            # No model pickle file right now
            return True
        else:
            with file.open(save_file_name, 'r') as f:
                with tempfile.NamedTemporaryFile(mode="w+b", suffix='.text') as temp_f:
                    # Write content to temp file
                    temp_f.write(f.read())
                    # Read temp file content
                    self.mdl.load_model(temp_f.name)
        return True

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        # Save Model
        if self.mdl is not None:
            with tempfile.NamedTemporaryFile(mode="w+b", suffix='.text') as temp_f:
                self.mdl.save_model(temp_f.name)
                save_file_name = "model.text"
                with file.open(save_file_name, 'w') as f:
                    temp_f.seek(0)
                    f.write(temp_f.read())

        return True

    def __call__(self, X, *args, target=True, index=False, **kwargs):
        raise NotImplementedError()


class ClassifierModel(Model):
    def __init__(self):
        self.cls = xgboost.XGBClassifier

    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl.predict_proba(X, *args, **kwargs)


class RegressionModel(Model):
    def __init__(self):
        self.cls = xgboost.XGBRegressor

    def __call__(self, X, *args, target=True, index=False, **kwargs):
        return self.mdl.predict(X, *args, **kwargs)
