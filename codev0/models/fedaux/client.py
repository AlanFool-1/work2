"""FedAux client ported from Fedrated's fedpub_aux implementation."""

from models.fedpub.client import Client as FedPubClient
from models.fedaux.nets import MaskedGCN


class Client(FedPubClient):
    def build_model(self):
        return MaskedGCN(
            self.args.n_feat,
            self.args.n_dims,
            self.args.n_clss,
            self.args.l1,
            self.args,
        )

    def extra_upload(self):
        # ``aux`` is already present in the uploaded model state. The server
        # reads it there instead of sending a duplicate array.
        return {}, 0
