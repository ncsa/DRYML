class Meta(type):
    # Since methods are part of the class, we only have to remove data from the object. We mark the protected data here. Keep up to date with attributes added 
    __orig_attrs__ = [
        '__args__',
        '__kwargs__',
        '__initialized__',
        '__locked__',
    ]

    @staticmethod
    def check_existing_attributes(obj):
        # Check if these attributes are already defined. Throw an error if they are.
        colliding_attrs = []
        for attr in Meta.__orig_attrs__:
            if hasattr(obj, attr):
                colliding_attrs.append(attr)
        if len(colliding_attrs) > 0:
            raise AttributeError(f"Attributes {colliding_attrs} already exist on object. Cannot create object.")

    def __call__(cls, *args, **kwargs):
        ## Initial creation
        obj = cls.__new__(cls)
        Meta.check_existing_attributes(obj)
        obj.__args__ = args
        obj.__kwargs__ = kwargs
        obj.__locked__ = False
        obj.__initialized__ = False
        return obj


class Object(metaclass=Meta):
    def __getattribute__(self, name):
        # First, check if we have this attribute
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # If we don't next check if we're initialized
            if not super().__getattribute__('__initialized__'):
                super().__getattribute__('__initialize__')()
        # Then check again
        return super().__getattribute__(name)
    
    def __initialize__(self):
        self.__init__(*self.__args__, **self.__kwargs__)
        self.__initialized__ = True

    def _unload(self):
        # Remove all attributes besides self._orig_attrs
        for attr in list(self.__dict__.keys()):
            if attr not in Meta.__orig_attrs__:
                delattr(self, attr)
        self.__initialized__ = False

    @property
    def definition(self):
        return {
            'cls': type(self),
            'args': self.__args__,
            'kwargs': self.__kwargs__, }
