from src.train._evo import EvosaxTrainer
from src.train._utils import pickle_save
from src.tasks._metaevo import MetaEvolutionTask
from src.tasks._utils import load_emoji

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

import evosax as ex

import emoji


W, H = 128, 128
TARGETS = [emoji.emojize(s) for s in [":lizard:"]]
TARGETS = jnp.stack([load_emoji(targ, W) for targ in TARGETS])

INNER_POPSIZE = 32
CROSSOVER_RATE = 0.1

OUTER_POPSIZE = 256
OUTER_GENS = 2_000
OUTER_REPS = 1

N_SEEDS = 1
RNG = jr.PRNGKey(101)

SAVE_FILE = "metanca.pickle"

def main():

	model = ...
	params, statics = model.partition()
	params_shaper = ex.ParameterReshaper(params)

	loss_fn = lambda s1, s2: jnp.square(s1.X, s2.X).mean()
	initializer = None
	inner_es = ex.SimpleGA(popsize=INNER_POPSIZE, num_dims=model.dna_size)
	inner_es_params = inner_es.default_params
	inner_es_params = inner_es_params.replace(
		cross_over_rate=CROSSOVER_RATE
	)
	task = MetaEvolutionTask(statics, TARGETS, loss_fn, initializer, inner_es, strategy_params=inner_es_params)
	
	outer_es = ex.DES(popsize=OUTER_POPSIZE)
	outer_es_params = outer_es.default_params
	train_fn = EvosaxTrainer(task, outer_es, outer_es_params, params_shaper, gens=OUTER_GENS, n_repeats=OUTER_REPS)

	if N_SEEDS > 1:
		best_params, es_states = jax.vmap(train_fn)(jr.split(RNG, N_SEEDS))
	else: 
		best_params, es_states = train_fn(RNG)

	pickle_save(SAVE_FILE, {"best_params": best_params, "data": es_states})


if __name__ == '__main__':
	#main()
	import matplotlib.pyplot as plt
	plt.imshow(TARGETS[0]); plt.show()

