class SaveCache(object):
    def __init__(self):
        self.save_object_cache = {}
        self.save_compute_cache = set()

    @property
    def obj_cache(self):
        return self.save_object_cache

    @property
    def compute_cache(self):
        return self.save_compute_cache

    def __del__(self):
        # Close int_files in save_cache
        for key in self.save_object_cache:
            self.save_object_cache[key].close()

    def __repr__(self):
        return f"object_cache: {self.save_object_cache} " \
           f"compute_cache: {self.save_compute_cache}"
