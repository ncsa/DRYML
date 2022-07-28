import io
from io import BufferedIOBase
import tempfile
import zipfile


class FileWriteIntermediary(BufferedIOBase):
    def __init__(self, mem_mode=False):
        self.mem_mode = mem_mode
        if mem_mode:
            self.tmp_file = io.BytesIO()
        else:
            self.tmp_file = tempfile.NamedTemporaryFile(mode='w+b')

    # Implement IOBase, and Buffered IOBase members
    def close(self):
        self.tmp_file.close()

    @property
    def closed(self):
        return self.tmp_file.closed

    def fileno(self):
        return self.tmp_file.fileno()

    def flush(self):
        return self.tmp_file.flush()

    def isatty(self):
        return self.tmp_file.isatty()

    def readable(self):
        return self.tmp_file.readable()

    def readline(self, size=-1):
        return self.tmp_file.readline(size)

    def readlines(self, hint=-1):
        return self.tmp_file.readlines(hint)

    def seek(self, offset, whence=io.SEEK_SET):
        return self.tmp_file.seek(offset, whence)

    def seekable(self):
        return self.tmp_file.seekable()

    def tell(self):
        return self.tmp_file.tell()

    def truncate(self, size=None):
        return self.tmp_file.truncate(size)

    def writable(self):
        return self.tmp_file.writable()

    def writelines(self, lines):
        return self.tmp_file.writelines(lines)

    def read(self, size=-1):
        return self.tmp_file.read(size)

    def read1(self, size=-1):
        return self.tmp_file.read1(size)

    def readinto(self, b):
        return self.tmp_file.readinto(b)

    def readinto1(self, b):
        return self.tmp_file.readinto1(b)

    def write(self, b):
        return self.tmp_file.write(b)

    def is_empty(self):
        old_pos = self.tell()
        self.seek(0)
        self.seek(0, 2)
        size = self.tell()
        empty = False
        if size != 0:
            with zipfile.ZipFile(
                    self, mode='r') as zf:
                if len(zf.namelist()) == 0:
                    empty = True
        self.seek(old_pos)
        return empty

    # Implement with statement magic methods
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    # My methods
    def write_to_file(self, file):
        # Save current file position
        cur_pos = self.tell()

        # Flush to disk, and Seek to beginning
        self.flush()
        self.seek(0)

        # Read current file content into file
        if type(file) is str:
            file = open(file, 'wb')

        file.write(self.read())

        # Restore position
        self.seek(cur_pos)
