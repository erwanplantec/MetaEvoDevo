from src.train._utils import progress_bar_scan

import jax
import jax.numpy as jnp
import jax.random as jr
import evosax as ex

from typing import Callable, Union, Tuple, Optional
from jaxtyping import PyTree

def EvosaxTrainer(eval_fn: Callable, 
                  strategy: Union[ex.Strategy, str],  
                  params_shaper: ex.ParameterReshaper, 
                  es_params: Optional[ex.EvoParams] = None,
                  gens: int = 100,
                  progress_bar: bool = True,
                  n_repeats: int = 1
                  )->Callable[[jr.PRNGKeyArray], Tuple[PyTree,dict]]:

    """Wrapper for evosax."""

    if isinstance(strategy, str):
        es = getattr(ex, strategy)
    else:
        es = strategy
    
    if es_params is None:
        es_params = es.default_params

    mapped_eval = jax.vmap(eval_fn, in_axes=(0, None))
    def _eval_fn(k, p):
        fits = mapped_eval(jr.split(k, n_repeats), p) #(nrep, pop)"
        avg = jnp.mean(fits, axis=0) #(pop,)
        return avg
         
    def evo_step(carry, x):
        key, es_state = carry
        key, ask_key, eval_key = jr.split(key, 3)
        flat_params, es_state = es.ask(ask_key, es_state, es_params)
        params = params_shaper.reshape(flat_params)
        fitness = jax.vmap(_eval_fn, in_axes=(None,0))(eval_key, params)
        es_state = es.tell(flat_params,
                             fitness,
                             es_state,
                             es_params)
        return [key, es_state], [es_state, fitness]

    if progress_bar: evo_step = progress_bar_scan(gens)(evo_step)

    def _train(key: jr.PRNGKeyArray, **init_kws):

        strat_key, evo_key = jr.split(key)
        es_state = es.initialize(strat_key, es_params)
        es_state = es_state.replace(**init_kws)
        [key, es_state], [es_states, fitnesses] = jax.lax.scan(
            evo_step, [evo_key, es_state],
            jnp.arange(gens)
        )
        return params_shaper.reshape_single(es_state.best_member), {"fitnesses": fitnesses, "states": es_states}

    return _train