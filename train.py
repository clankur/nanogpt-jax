import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import json
from clearml import Task
from typing import Optional, List, Tuple
from jaxtyping import PyTree 
from datetime import datetime

from model import GPTLanguageModel, encode, decode, train_data, val_data, block_size, vocab_size

hyperparams = {
    "block_size": 256,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "vocab_size": vocab_size
}

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
            _, loss, _ = jax.vmap(model)(X, Y) 
            losses = jax.ops.index_update(losses, k, loss)
        out[split] = losses.mean()
    return out

def make(*, key: jnp.ndarray, block_size: int, n_embd: int, n_head: int, n_layer: int, vocab_size: int) -> GPTLanguageModel:
    return GPTLanguageModel(jr_key=key, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, vocab_size=vocab_size)

def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make(key=jax.random.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

if __name__ == "__main__":
    # Get the current date and time
    current_date_time = datetime.now()

    # Format the date and time in a string
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_date_time)

    # print("initializing the task")
    # task = Task.init(project_name='nanogpt', task_name=formatted_date_time)
    # task.execute_remotely('default', clone=False, exit_process=True)

    print("training the model")
    model = GPTLanguageModel(key=jax.random.PRNGKey(0), **hyperparams)
    print("model created")

    # create a jax optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)
    print("optimizer created")

    xb, yb = get_batch('train')
    print(xb.shape)

    @eqx.filter_jit
    def update(model, opt_state, xb, yb):
        logits, loss, grads = jax.vmap(model)(xb, yb) 
        # compute gradients

        # apply updates based on grads
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates) # update model
        return loss, model, opt_state

    print("starting training")
    # training the model
    for steps in range(max_epochs):
        # every once in a while eval loss on train and val sets
        if steps % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Step: {steps}, Train loss: {losses['train']:.2f}, Val loss: {losses['val']:.2f}")   

            # task.get_logger().report_scalar(title="losses", series="train", value=losses['train'], iteration=steps)
            # task.get_logger().report_scalar(title="losses", series="val", value=losses['val'], iteration=steps)
        
        xb, yb = get_batch('train')
        loss, model, opt_state = update(model, opt_state, xb, yb)

        # task.get_logger().report_scalar(title="losses", series="train", value=loss, iteration=steps)
    
    save("gpt_model.eqx", hyperparams, model)