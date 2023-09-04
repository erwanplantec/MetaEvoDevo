import jax
import jax.random as jr
import equinox as eqx

from typing import NamedTuple, Tuple
from jaxtyping import PyTree

class State(NamedTuple):
	dna: PyTree[...]

class DevoModel(eqx.Module):

	#-------------------------------------------------------------------

	def __call__(self, state: State,  key: jr.PRNGKeyArray)->State:
		
		raise NotImplementedError()

	#-------------------------------------------------------------------

	@property
	def dna_size(self):
		"""Size of the DNA vector"""
		raise NotImplementedError()

	def partition(self):
		"""Define how the model should partition between params and statics"""
		return eqx.partition(self, eqx.is_array)

	#-------------------------------------------------------------------

	def rollout(self, init_sate: State, key: jr.PRNGKeyArray, n: int)->Tuple[State, State]:

		def _step(c, x):
			s, k = c
			k, sk = jr.split(k)
			ns = self.__call__(s, sk)
			return [ns, k], ns

		return jax.lax.scan(_step, [init_sate, key], None, n)

	#-------------------------------------------------------------------

