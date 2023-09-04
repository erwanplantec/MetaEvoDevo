from src.train._evo import EvosaxTrainer
from src.train._utils import pickle_save
from src.tasks._metaevo import MetaEvolutionTask
from src.tasks._utils import load_emoji
from src.models._nca import NCA, NCAState

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

import evosax as ex

import emoji


W, H, C = 8, 8, 5
PERCEPTION_FTS = 1

TARGETS = [emoji.emojize(s) for s in [":lizard:"]]
TARGETS = jnp.stack([load_emoji(targ, W).transpose([2,0,1]) for targ in TARGETS])

INNER_POPSIZE = 2
CROSSOVER_RATE = 0.1

OUTER_POPSIZE = 4
OUTER_GENS = 3
OUTER_REPS = 1

N_SEEDS = 1
RNG = jr.PRNGKey(101)

SAVE_FILE = "metanca.pickle"

def main():

	key_train, key_model = jr.split(RNG, 2)

	model = NCA(C, PERCEPTION_FTS, key=key_model, update_features=[2])
	params, statics = model.partition()
	params_shaper = ex.ParameterReshaper(params)

	loss_fn = lambda state, goal: jnp.square(state.X[:4] - goal).mean()
	initializer = lambda key, dna: NCAState(
		dna=dna, 
		X=jnp.zeros((INNER_POPSIZE,C,H,W)).at[:, H//2, W//2].set(1.)
	)
	inner_es = ex.SimpleGA(popsize=INNER_POPSIZE, num_dims=model.dna_size)
	inner_es_params = inner_es.default_params
	inner_es_params = inner_es_params.replace(
		cross_over_rate=CROSSOVER_RATE
	)
	task = MetaEvolutionTask(statics, TARGETS, loss_fn, initializer, inner_es, strategy_params=inner_es_params)
	
	outer_es = ex.DES(popsize=OUTER_POPSIZE, num_dims=params_shaper.total_params)
	outer_es_params = outer_es.default_params
	train_fn = EvosaxTrainer(task, outer_es, outer_es_params, params_shaper, gens=OUTER_GENS, n_repeats=OUTER_REPS)
	train_fn = eqx.filter_jit(train_fn)

	if N_SEEDS > 1:
		best_params, es_states = jax.vmap(train_fn)(jr.split(key_train, N_SEEDS))
	else: 
		best_params, es_states = train_fn(key_train)

	pickle_save(SAVE_FILE, {"best_params": best_params, "data": es_states})


if __name__ == '__main__':
	main()
