class PositionalLaplacianEncoding(object):
    def __init__(self, k=2):
        self.k = k

    def __call__(self, data):
        if self.k == 0:
            return data
        else:
            num_nodes = data.x.shape[0]

            L = get_laplacian(
                data.edge_index, normalization="sym", num_nodes=num_nodes)

            L = torch.sparse.FloatTensor(
                L[0], L[1], size=(num_nodes, num_nodes)).to_dense()

            EigVal, EigVec = torch.eig(L, eigenvectors=True)
            idx = EigVal[:, 0].argsort()
            ordered_eigvec = EigVec[idx]
            pos_enc = ordered_eigvec[:, :self.k]

            data["pos_enc"] = pos_enc

            return data