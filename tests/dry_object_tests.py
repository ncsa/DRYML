import dryml

def test_basic_object_1():
    # Define simple class
    class SimpleObject(dryml.DryObject):
        def __init__(self, i=0):
            self.i = i
            dry_kwargs = {
                'i': i
            }
            super().__init__(
                dry_kwargs=dry_kwargs
            )

        def __equal__(rhs):
            return self.i == rhs.i

    temp_buffer = io.BytesIO(b'')
    obj = SimpleObject(10)

    # Test that save to buffer works
    assert obj.save_self(temp_buffer)

    obj2 = dryml.load_object(temp_buffer)

    # Test that restore from buffer creates identical object in this context.
    assert obj == obj2

