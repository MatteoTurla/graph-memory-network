import torch
from collections import OrderedDict
from model.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from torch_geometric.utils import add_self_loops, to_undirected

class GraphMemoryNetwork(torch.nn.Module):
	def __init__(self, input_size, n_heads, n_layers, n_class):
		super().__init__()
		
		self.layers = self.heads = torch.nn.ModuleList(
				[MultiHeadAttentionLayer(input_size, n_heads) for i in range(n_layers)]
				)

		self.feed_forward = torch.nn.Sequential(OrderedDict([
			('end_1', torch.nn.Linear(input_size, 512)),
			('relu1', torch.nn.ReLU()),
			('end_2', torch.nn.Linear(512, 512)),
			('relu2', torch.nn.ReLU()),
			('end_3', torch.nn.Linear(512, n_class))
			]))


	def forward(self, X, edge_index):

		for layer in self.layers:
			X = layer(X, edge_index)

		y_hat = self.feed_forward(X)

		return y_hat

	def inference(self, x_all, subgraph_loader, device):
		# used to evaluate the model in a faster and memory efficient way
		for layer in self.layers:
			xs = []
			for batch_size, n_id, adj in subgraph_loader:

				edge_index, _, _ = adj
				# we are using neighbours sampler that return a direct bipartite graph
				edge_index = to_undirected(edge_index)
				edge_index = edge_index.to(device)
				
				x = x_all[n_id].to(device)
				x = layer(x, edge_index)[:batch_size]

				xs.append(x)

			x_all = torch.cat(xs, dim=0)

		# now feed forward for the final classification
		y_hat = self.feed_forward(x_all)

		return y_hat


