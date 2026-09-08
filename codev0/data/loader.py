from misc.utils import *

class DataLoader:
    def __init__(self, args):
        self.args = args
        # Each client partition contains exactly one PyG Data object.  The
        # original value 1 repeatedly spawned nested loader subprocesses from
        # every federated worker for train/val/test.  Zero preserves identical
        # data order and batching while avoiding that pure process overhead.
        self.n_workers = int(getattr(args, 'loader_workers', 0))
        self.client_id = None

        from torch_geometric.loader import DataLoader
        self.DataLoader = DataLoader

    def switch(self, client_id):
        if not self.client_id == client_id:
            self.client_id = client_id
            self.partition = get_data(self.args, client_id=client_id)
            self.pa_loader = self.DataLoader(dataset=self.partition, batch_size=1, 
                shuffle=False, num_workers=self.n_workers, pin_memory=False)

def get_data(args, client_id):
    return [
        torch_load(
            args.data_path, 
            f'{args.dataset}_{args.mode}/{args.n_clients}/partition_{client_id}.pt'
        )['client_data']
    ]
