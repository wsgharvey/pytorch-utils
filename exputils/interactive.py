import functools as ft
import itertools as it
import numpy as np

from .display import display_level_decorator, display
from .db import retrieve_from_database


class PrintableTable():

    def __init__(self, headers, item_width=6):

        self.item_width = item_width
        self.body = ""
        self.add_multiline_row(headers)
        self.body += "\n"

    def add_multiline_row(self, l):
        # spills over to multiple lines to write full contents

        def get_cell(item):
            item = str(item)
            return [item[i:i+self.item_width]
                    for i in range(0, len(item),
                                   self.item_width)]
        def get_row(cell, r):
            if len(cell) <= r:
                return ' '*self.item_width
            else:
                return cell[r] + \
                       ' '*(self.item_width-len(cell[r]))
        cells = [get_cell(item) for item in l]
        n_rows = max(len(cell) for cell in cells)
        for r in range(n_rows):
            self.body += " ".join(get_row(cell, r)
                                  for cell in cells)+'\n'

    def add_row(self, l):

        def format(item):
            item = str(item)[:self.item_width]
            return item + " "*(self.item_width-len(item))
        self.body += " ".join(format(item) for item in l)+'\n'

    def print(self):

        display("user-requested", self.body)


class ExperimentTracker():
    """
    must have a self.table_getter attribute provided e.g. by subclass

    In the below:
     - `num` generally refers to group of runs (e.g. all with identical
         args except for random seed)
     - `ID` generally refers to a single run
    """

    def __init__(self, *args, **kwargs):

        self.print_summary(*args, **kwargs)

    def print_summary(self, **restrictions):

        with self.table_getter as table:
            relevant = [field for field in table.columns
                        if field not in self.irrelevant]
            run_types = list(table.distinct(*relevant))
            self.run_types = [settings for settings in run_types if
                              all(settings[kw] == restrictions[kw]
                                  for kw in restrictions.keys())]
            to_print = PrintableTable(['ID']+list(relevant)+['No. runs'])
            for num, settings in enumerate(self.run_types):
                n_runs = len(list(table.find(**settings)))
                to_print.add_row([num]+list(settings.values()) + [n_runs])
        to_print.print()

    def get_from_id(self, ID):

        with self.table_getter as table:
            return table.find(ID=ID)

    def get_from_num(self, num):

        run_type = self.run_types[num]
        with self.table_getter as table:
            return list(table.find(**run_type))

    @display_level_decorator(False)
    def print_details(self, *nums):

        for num in nums:

            runs = self.get_from_num(num)
            to_print = PrintableTable(
                list(runs[0].keys())+
                ['ep. so far', ('valid' if self.has_valid else 'loss')])

            for run in runs:

                net = type(self).init_from_args(run)
                net.load_checkpoint()
                n_epochs = net.epochs
                if self.has_valid:
                    if len(net.losses['valid']) > 0:
                        loss = min(net.losses['valid'])
                    else:
                        loss = 'N/A'
                else:
                    if len(net.losses['train']) == 0:
                        loss = 'N/A'
                    else:
                        recent = net.losses['train'][-1000:]
                        loss = sum(recent)/len(recent)

                to_print.add_row(
                    list(run.values())+[n_epochs, loss]
                )
            to_print.print()
