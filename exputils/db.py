import git
import dataset as pds
from .display import display
import time


database_url = 'sqlite:///experiments.db'


class TableGetter():

    def __init__(self, path_or_url, location, tablename):

        self.url = {
            'path': f'sqlite:///{location}',
            'url': location,
        }[path_or_url]
        self.tablename = tablename

    def __enter__(self):

        db = pds.connect(self.url)
        return db[self.tablename]

    def __exit__(self, *args):

        return


def get_git_hash():

    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

def get_git_time():

    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.committed_date

def try_multiple(foo):

    def wrapped(*args, **kwargs):
        for i in range(10):
            try:
                return foo(*args, **kwargs)
            except Exception as err:
                display("error", f"Hit exception "+\
                        f"{type(err).__name__} with {args}, {kwargs}")
                time.sleep(1)
        display("error", "Timed out.")
        raise Exception

    return wrapped

# @try_multiple   # TODO I don't know if this wrapper helps at all
def add_to_database(table_getter, entry_dict):

    entry_dict['git_commit'] = get_git_hash()
    entry_dict['git_time'] = get_git_time()
    with table_getter as table:

        # check we don't have run already
        maybe_unique = entry_dict.copy()
        for field in ["git_commit", "git_time"]:
            maybe_unique.pop(field)
        in_database = len(list(table.find(**maybe_unique))) > 0
        if in_database:
            display("info", "Matching entry already in database.")
            return

        table.insert(entry_dict)

# @try_multiple
def retrieve_from_database(table_getter, table_id):

    with table_getter as table:

        result = next(table.find(id=table_id))
        if result is None:
            raise Exception("Table lookup returned None.")
        return result
