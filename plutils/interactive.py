from exputils.interactive import ExperimentTracker


class TrainableTracker(ExperimentTracker):

    table_getter = None
    init_from_args = None

    def load_from_id(self, ID):

        return self.load_from_args(
            self.load_from_id(ID)
        )

    def load_from_args(self, args,
                       n_epochs=None,
                       time=None,
                       is_best=False,
                       is_latest=False):

        # check only one option specified
        is_epochs = n_epochs is not None
        is_timed = time is not None
        assert sum([is_epochs, is_timed, is_best, is_latest]) == 1

        net = self.init_from_args(args)
        if is_epochs:
            net.load_checkpoint(max_epochs=n_epochs)
            assert net.epochs == n_epochs
        elif is_timed:
            net.load_timed_checkpoint(time)
        elif is_best:
            net.load_best_checkpoint()
        elif is_latest:
            net.load_checkpoint()
        return net

    def iter_networks(self, num):
        """
        Useful for e.g. plotting stuff over the course of training
        """
        argses = self.get_from_num(num)
        for args in argses:

            net = self.load_from_args(args, is_latest=True)
            yield net

