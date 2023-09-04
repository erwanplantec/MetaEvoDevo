from src.models._base import DevoModel, State
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

import equinox as eqx 
import equinox.nn as nn

from typing import Callable, Optional, NamedTuple, Any, Union
from jaxtyping import Float, Array, PyTree


class Node(NamedTuple):
    p: Float[Array, "N dims"]
    h: Float[Array, "N features"]
    mask: Optional[Float[Array, "N"]]=None

class Edge(NamedTuple):
    receivers: jax.Array
    senders: jax.Array
    e: Optional[jax.Array]=None
    mask: Optional[jax.Array]=None  

class Graph(State):
    nodes: Node
    edges: Edge
    infos: Optional[Any]=None


class KNNConnector(eqx.Module):

    """Create edges between to a node from its k nearest neighbors"""
    #-------------------------------------------------------------------
    k: int
    #-------------------------------------------------------------------

    def __init__(self, k: int = 5):
        
        self.k = k

    #-------------------------------------------------------------------

    def __call__(self, graph: Graph, key: jr.PRNGKeyArray):

        p = graph.nodes.p
        max_nodes = p.shape[0]
        dp = p[:, None, :] - p
        d = (dp*dp).sum(-1)
        _, idxs = jax.lax.top_k(-d, self.k)

        s = jnp.where(graph.active_nodes[:, None], idxs[:, :self.k], max_nodes-1)
        r = jnp.where(graph.active_nodes[:, None], jnp.mgrid[:max_nodes, :self.k][0], max_nodes-1)

        s = s.reshape((-1,))
        r = r.reshape((-1,))

        return graph._replace(edges=graph.edges._replace(senders=s, receivers=r))

    #-------------------------------------------------------------------


class PGNN(DevoModel):

    """Particle Graph Neural Network"""
    #-------------------------------------------------------------------
    rnn: eqx.Module
    msg_fn: eqx.Module
    policy_fn: eqx.Module
    connector: Union[eqx.Module, Callable]
    aggr_fn: Callable
    spatial_encoder: Callable
    #-------------------------------------------------------------------

    def __init__(
        self, 
        node_features: int, 
        msg_features: int, 
        *, 
        key: jr.PRNGKeyArray,
        cell_type=nn.GRUCell, 
        connector: Union[eqx.Module, Callable]=KNNConnector(k=5),
        aggr_fn=jax.ops.segment_sum,
        spatial_encoder: Callable=lambda x:x, 
        spatial_encoding_dims: int=2):

        key, k1, k2, k3 = jr.split(key, 4)
        self.rnn = cell_type(msg_features+spatial_encoding_dims, node_features, key=k1)
        self.msg_fn = nn.MLP(node_features+2, msg_features, 16, 1, key=k2, final_activation=jnn.relu)
        self.policy_fn = nn.MLP(node_features, 2, 16, 1, key=k3)
        self,connector = connector
        self.aggr_fn = aggr_fn
        self.spatial_encoder = spatial_encoder

    #-------------------------------------------------------------------

    def __call__(self, graph: Graph, key: jr.PRNGKeyArray):
        """Summary
        
        Args:
            graph (Graph): Description
            key (jr.PRNGKeyArray): Description
        
        Returns:
            TYPE: Description
        """
        graph = self.connector(graph, key)
        max_nodes = graph.nodes.h.shape[0]
        p, h = graph.nodes.p, graph.nodes.h
        dp = p[:, None, :] - p #(n,n,2)
        r = dp[graph.edges.receivers, graph.edges.senders]
        m = jax.vmap(self.msg_fn)(jnp.concatenate([h[graph.edges.senders], r], axis=-1))
        aggr_m = self.aggr_fn(m, graph.edges.receivers, max_nodes)
        p_enc = jax.vmap(self.spatial_encoder)(p)
        nh = jax.vmap(self.rnn)(jnp.concatenate([aggr_m, p_enc], axis=-1), h)
        v = jax.vmap(self.policy_fn)(nh)
        v = jnp.clip(v, -1., 1.)*.1

        nodes = Node(h=nh, p=p+v)

        return graph._replace(nodes=nodes)

    #-------------------------------------------------------------------



@jax.jit
def incr(x):
    n = x.sum().astype(int)
    return x.at[n].set(1.)

@jax.jit
def nincr(x, d):
    return jax.lax.fori_loop(0, d.sum().astype(int), lambda i, x: incr(x), x)

class GPGNN(DevoModel):

    """Growing Paryicle Graph Neural Network"""
    #-------------------------------------------------------------------
    rnn: eqx.Module
    msg_fn: eqx.Module
    policy_fn: eqx.Module
    aggr_fn: Callable
    spatial_encoder: Callable
    alpha: float
    stochastic_div: bool
    #-------------------------------------------------------------------

    def __init__(
        self, 
        node_features: int, 
        msg_features: int, 
        *, 
        key: jr.PRNGKeyArray,
        cell_type=nn.GRUCell, 
        aggr_fn=jax.ops.segment_sum,
        spatial_encoder: Callable=lambda x:x, 
        spatial_encoding_dims: int=2,
        alpha: float=.1, 
        stochastic_div: bool=True):

        key, k1, k2, k3 = jr.split(key, 4)
        self.rnn = cell_type(msg_features+spatial_encoding_dims, node_features, key=k1)
        self.msg_fn = nn.MLP(node_features+2, msg_features, 32, 1, key=k2, final_activation=jnn.relu)
        self.policy_fn = nn.MLP(node_features, 2, 32, 1, key=k3)
        self.aggr_fn = aggr_fn
        self.spatial_encoder = spatial_encoder
        self.alpha = alpha
        self.stochastic_div = stochastic_div

    #-------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, graph: Graph, key: jr.PRNGKeyArray):

        assert graph.nodes.mask is not None

        key, kd, kr = jr.split(key, 3)
        max_nodes = graph.nodes.h.shape[0]
        p, h, mask = graph.nodes.p, graph.nodes.h, graph.nodes.mask
        dp = p[:, None, :] - p #(n,n,2)
        r = dp[graph.receivers, graph.senders]
        m = jax.vmap(self.msg_fn)(jnp.concatenate([h[graph.senders], r], axis=-1))

        aggr_m = self.aggr_fn(m, graph.receivers, max_nodes)
        p_enc = jax.vmap(self.spatial_encoder)(p)

        nh = jax.vmap(self.rnn)(jnp.concatenate([aggr_m, p_enc], axis=-1), h)
        v = jax.vmap(self.policy_fn)(nh)
        v = jnp.clip(v, -1., 1.)*.1

        if self.stochastic_div:
            pd = jnn.sigmoid(10. * (nh[:,0]-self.alpha))
            d = jnp.where(jr.uniform(kr, (max_nodes,), minval=0., maxval=1.)<pd, 1., 0.)
        else:
            d = jnp.where(nh[:, 0]>self.alpha, 1., 0.)

        d = jnp.where(graph.nodes.mask, d, 0.)

        nmask, nh, np = jax.lax.cond(d.sum()>0., 
                                     lambda *args: self._add_nodes(*args),
                                     lambda *_: (mask, nh, p),
                                     d, nh, p, mask, kd)

        nodes = Node(h=nh,
                     p=np+jnp.where(mask[:,None], v, 0.0),
                     mask=nmask)

        return graph._replace(nodes=nodes)

    #-------------------------------------------------------------------

    def _add_nodes(self, d, h, p, mask, key):

        nmask = nincr(mask, d)
        max_nodes = h.shape[0]

        tgt = jnp.cumsum(d) * d - d
        tgt = jnp.where(d, tgt.astype(int), -1) + mask.sum().astype(int) * d.astype(int)
        mask_new = nmask * (1.-mask)

        nh = jnp.where(mask_new[:, None], jnp.zeros(h.shape), h)
        nh = jnp.where(d[:, None], nh.at[:, 0].set(0.), nh)

        np = jax.ops.segment_sum(p, tgt, max_nodes)
        np = np + (jr.normal(key, p.shape) * .001)
        np = jnp.where(mask_new[:, None], np, p)

        return nmask, nh, np


