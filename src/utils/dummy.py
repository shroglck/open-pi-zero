import torch


class NoSyncBase:
    def no_sync(self):
        if self.use_ddp:
            # If DDP is used, call the actual `no_sync` method
            return torch.nn.parallel.DistributedDataParallel.no_sync(self)
        else:
            # Otherwise, return the dummy context manager
            class DummyContext:
                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_value, traceback):
                    pass

            return DummyContext()
