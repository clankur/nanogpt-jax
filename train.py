import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from flax import serialization

from clearml import Task
from typing import Optional, List, Tuple
from jaxtyping import PyTree 
import datetime

from model import GPTLanguageModel, encode, decode, train_data, val_data, block_size

batch_size = 64
learning_rate = 3e-4
max_epochs = 5000
eval_interval = 500
eval_iters = 200

def get_batch(split: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    data = train_data if split == 'train' else val_data
    ix = jax.random.randint(jax.random.PRNGKey(0), (batch_size,), 0, len(data) - block_size) # will return batch_size random numbers that are offsets of the data set 
    x = jnp.stack([data[i:i+block_size] for i in ix]) # builds a stack of tensors of size blocksize for each random number in ix
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 stack of tensors
    return x, y

def estimate_loss(model) -> dict:
    out = {}
    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses = jax.ops.index_update(losses, k, loss)
        out[split] = losses.mean()
    return out

if __name__ == "__main__":
        
    # Get the current date and time
    current_date_time = datetime.now()

    # Format the date and time in a string
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")
    task = Task.init(project_name='nanogpt', task_name=formatted_date_time)
    task.execute_remotely('default', clone=False, exit_process=True)

    model = GPTLanguageModel()
    
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, block_size)))

    # create a jax optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, xb, yb):
        logits, loss = model.apply(params, xb, yb)
        # compute gradients
        grads = jax.grad(lambda p, x, y: model.apply(p, x, y)[1])(params, xb, yb) # compute gradients

        # apply updates based on grads
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates) # update parameters
        return params, opt_state, loss

    # training the model
    for steps in range(max_epochs):
        # every once in a while eval loss on train and val sets
        if steps % eval_interval == 0:
            losses = estimate_loss(model)
            task.get_logger().report_scalar(title="losses", series="train", value=losses['train'], iteration=steps)
            task.get_logger().report_scalar(title="losses", series="val", value=losses['val'], iteration=steps)
        
        xb, yb = get_batch('train')
        params, opt_state, loss = update(params, opt_state, xb, yb)

        task.get_logger().report_scalar(title="losses", series="train", value=loss, iteration=steps)
        
    with open('model_weights.msgpack', 'wb') as f:
        f.write(serialization.to_bytes(params))