from src.train._utils import progress_bar_scan

import jax
import jax.numpy as jnp
import jax.random as jr
import evosax as ex
from typing import Callable, Optional, Tuple
from jaxtyping import PyTree

def EvosaxTrainer(eval_fn: Callable, 
                  strategy: ex.Strategy, 
                  es_params: ex.EvoParams, 
                  params_shaper: ex.ParameterReshaper, 
                  gens: int = 100,
                  progress_bar: bool = True,
                  plot: bool = False,
                  n_repeats: int = 1,
                  var_penalty: float = 0.)->Callable[[jr.PRNGKeyArray,], Tuple[PyTree[...], PyTree[...]]]:

    """Wrapper for evosax."""
    if n_repeats > 1:
        mapped_eval = jax.vmap(eval_fn, in_axes=(0, None))

        def eval_fn(k, p):
            fits = mapped_eval(jr.split(k, n_repeats), p) #(nrep, pop)"
            avg = jnp.mean(fits, axis=0) #(pop,)
            var = jnp.var(fits, axis=0) #(pop,)
            return avg - var*var_penalty
         
    def evo_step(carry, x):
        key, es_state = carry
        key, ask_key, eval_key = jr.split(key, 3)
        flat_params, es_state = strategy.ask(ask_key, es_state, es_params)
        params = params_shaper.reshape(flat_params)
        fitness = eval_fn(eval_key, params)
        es_state = strategy.tell(flat_params,
                             fitness,
                             es_state,
                             es_params)
        return [key, es_state], es_state

    if progress_bar: evo_step = progress_bar_scan(gens)(evo_step)
    #if plot: evo_step = plot_scan(evo_step, getter=lambda x, y: (x, x, y[1].min()))

    def _train(key: jr.PRNGKey, **init_kws):

        strat_key, evo_key = jr.split(key)
        es_state = strategy.initialize(strat_key, es_params)
        es_state = es_state.replace(**init_kws)
        [_, es_state], es_states = jax.lax.scan(
            evo_step, [evo_key, es_state],
            jnp.arange(gens)
        )
        best_params = params_shaper.reshape_single(es_state.best_member)
        return best_params, es_states

    return _train