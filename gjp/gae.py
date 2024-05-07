from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .decoder import GraphDecoder
from .model import MessagePassing


class GAE(nn.Module):
    encoder_stack: Sequence[Sequence[int]]
    max_num_nodes: int
    max_num_edges: int
    init_stack: Sequence[int]
    init_features: int
    prob_stack: Sequence[Sequence[int]]
    feature_stack: Sequence[Sequence[int]]
    node_features: int
    edge_features: int
    max_num_graph: int = None

    def setup(self):
        self.encoder = MessagePassing(node_feature_sizes=self.encoder_stack, edge_feature_sizes=self.encoder_stack, global_feature_sizes=self.encoder_stack, num_nodes=self.max_num_nodes * self.max_num_graph if self.max_num_graph else None)

        self.sigma_encoder = MessagePassing(node_feature_sizes=self.encoder_stack, edge_feature_sizes=self.encoder_stack, global_feature_sizes=self.encoder_stack, num_nodes=self.max_num_nodes * self.max_num_graph if self.max_num_graph else None)

        self.decoder = GraphDecoder(
            init_edge_stack=self.init_stack,
            init_edge_features=self.init_features,
            init_node_stack=self.init_stack,
            init_node_features=self.init_features,
            prob_node_stack=self.prob_stack,
            prob_edge_stack=self.prob_stack,
            feature_edge_stack=self.feature_stack + ((self.node_features,),),
            feature_node_stack=self.feature_stack + ((self.edge_features,),),
            mean_instead_of_sum=True,
            max_num_nodes=self.max_num_nodes,
            max_num_edges=self.max_num_edges,
        )

    def __call__(self, x):
        mu_wo_node = self.encoder(x).globals
        log_sigma_wo_node = self.sigma_encoder(x).globals
        sigma_wo_node = jnp.exp(log_sigma_wo_node)
        rng = self.make_rng("reparametrize")
        eps = jax.random.normal(rng, mu_wo_node.shape)
        z_wo_node = mu_wo_node + sigma_wo_node * eps  # Reparameterization trick

        # Add the node and edge info after adding noise
        z = jnp.vstack((z_wo_node.transpose(), x.n_node, x.n_edge)).transpose()
        reconstructed = self.decoder(z)
        return reconstructed

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def loss_function(params, in_graphs, rng, model, metric_params, metric_model, norm):
    in_metric = metric_model.apply(metric_params, in_graphs).globals[:-1]
    tmp_out_graphs = model.apply(params, in_graphs, rngs={"reparametrize": rng})
    embedded_space = tmp_out_graphs.globals[::2]
    embedded_space = embedded_space[:, :-2]

    out_globals = jnp.repeat(in_graphs.globals, 2, axis=0)
    out_graphs = tmp_out_graphs._replace(globals=out_globals)
    out_metric = metric_model.apply(metric_params, out_graphs).globals[::2]
    out_metric = out_metric[:-1]

    loss = jnp.mean(jnp.sqrt((in_metric - out_metric) ** 2))
    if norm:
        loss += jnp.mean(jnp.sqrt(embedded_space**2))

    return loss
