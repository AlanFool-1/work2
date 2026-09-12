"""FedAux server ported from Fedrated's fedpub_aux implementation."""

from models.fedpub.server import Server as FedPubServer
from models.fedaux.nets import MaskedGCN


class Server(FedPubServer):
    embedding_key = 'aux'
    uses_proxy = False

    def build_model(self):
        return MaskedGCN(
            self.args.n_feat,
            self.args.n_dims,
            self.args.n_clss,
            self.args.l1,
            self.args,
        )

    def personalization_embedding(self, message):
        return message['model']['aux']
