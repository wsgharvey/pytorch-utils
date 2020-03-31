def set_display(info=True):

    global _print_info
    _print_info = True

set_display()   # set with default values

def display(level, *args, **kwargs):

    assert level in [
        "info", "error", "user-requested"
    ]
    global _print_info
    if level == "info" and not _print_info:
        return

    print(*args, **kwargs)


class display_level():

    def __init__(self, print_info):

        self.inside = print_info

    def __enter__(self):

        global _print_info
        self.outside = _print_info
        _print_info = self.inside

    def __exit__(self, *args, **kwargs):

        global _print_info
        _print_info = self.outside


class display_level_decorator():

    def __init__(self, *args, **kwargs):
        self.display_args = args
        self.display_kwargs = kwargs

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):

            with display_level(*self.display_args,
                               **self.display_kwargs):

                return f(*args, **kwargs)

        return wrapped_f

