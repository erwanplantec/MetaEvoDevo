from src.models._base import DevoModel

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

from jaxtyping import Float, Array, Bool
from typing import Iterable, NamedTuple

import equinox as eqx
import equinox.nn as nn

class NCAState(NamedTuple):
	dna: Float[Array, "C"]
	X: Float[Array, "C H W"]


class PerceptionNetwork(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	conv: eqx.Module
	#-------------------------------------------------------------------

	def __init__(self, channels: int, perception_dims: int, *, key: jr.PRNGKeyArray):

		self.conv = nn.Conv2d(channels, perception_dims*channels,
							  kernel_size=3, stride=1, padding=1, 
							  groups=channels, key=key)

	#-------------------------------------------------------------------

	def __call__(self, X: Float[Array, "C H W"]):
		
		return self.conv(X)


class UpdateNetwork(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	layers: list
	#-------------------------------------------------------------------

	def __init__(self, features: Iterable[int], *, key: jr.PRNGKeyArray):
		
		self.layers = [
			nn.Conv2d(features[i-1], features[i], kernel_size=1, stride=1, padding=0, use_bias=i<len(features)-1, key=key)
		for i in range(1, len(features))] 

	#-------------------------------------------------------------------

	def __call__(self, X: Float[Array, "C H W"]):
		
		for layer in self.layers[:-1]:
			X = jnn.relu(layer(X))
		return self.layers[-1](X)


class NCA(DevoModel):
	
	"""
	"""
	#-------------------------------------------------------------------
	perception_net: PerceptionNetwork
	update_net: UpdateNetwork
	channels: int
	alpha: float
	#-------------------------------------------------------------------

	def __init__(self, channels: int, perception_dims: int, *, key:jr.PRNGKeyArray, update_features: Iterable=[128], alpha: float=.1):

		key, key_perc, key_upd = jr.split(key, 3)
		self.perception_net = PerceptionNetwork(channels, perception_dims, key=key_perc)
		self.update_net = UpdateNetwork([perception_dims*channels]+list(update_features)+[channels],
										key=key_upd)
		self.channels = channels
		self.alpha = alpha

	#-------------------------------------------------------------------

	@property
	def dna_size(self):
		return self.channels
		
	#-------------------------------------------------------------------

	def __call__(self, state: NCAState, *args):
		
		dna, X = state.dna, state.X
		life_mask = self._life_mask(X)
		X = X+(dna[:, None, None] * life_mask.astype(float))
		percept = self.perception_net(X)
		dX = self.update_net(percept)
		X = X + dX
		life_mask = life_mask & (self._life_mask(X))
		life_mask = life_mask.astype(float)
		X = X * life_mask

		return NCAState(X=X, dna=dna)

	#-------------------------------------------------------------------

	def _life_mask(self, X: Float[Array, "1 H W"])->Bool[Array, "1 H W"]:

		return nn.MaxPool2d(3, 1, 1)(X[-1:, ...]) > self.alpha



if __name__ == '__main__':
	from src.models._utils import render_vid
	import matplotlib.pyplot as plt

	W = H = 64
	C = 16
	nca = NCA(C, 3, key=jr.PRNGKey(1010101))

	s0 = NCAState(
		dna = jnp.ones((C,)),
		X = jnp.zeros((C, W, H)).at[:, W//2, H//2].set(.5)
	)

	_, states = nca.rollout(s0, jr.PRNGKey(101), 100)
	
	anim = render_vid(jnp.clip(states.X[:, [0, 1, 2, -1], ...].transpose([0,2,3,1]), 0., 1.))
	plt.show()








