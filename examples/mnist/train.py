from datasets import load_dataset
import optax
from flax import nnx
from functools import partial
import grain.python as grain
import mlflow
from tqdm import tqdm

ds = load_dataset("ylecun/mnist")

class GrainDataSource(grain.RandomAccessDataSource):
    def __init__(self, ds):
        self.ds = ds.with_format("jax")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


num_epochs = 2
batch_size = 32
eval_every = 10

index_sampler = grain.IndexSampler(
    len(ds["train"]),
    seed=2048,
    shuffle=True,
    shard_options=grain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
)

data_loader = grain.DataLoader(
    data_source=GrainDataSource(ds["train"]),
    operations=[grain.Batch(batch_size=batch_size)],
    sampler=index_sampler,
    worker_count=0
)

data_iter = iter(data_loader)

class CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Instantiate the model.
model = CNN(rngs=nnx.Rngs(0))

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: CNN, batch):
    logits = model(batch["image"][..., None])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("mnist")

with mlflow.start_run():

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in tqdm(enumerate(data_iter)):
        # for batch in tqdm(ds["train"].with_format("jax").iter(batch_size=batch_size)):
            # Run the optimization for one step and make a stateful update to the following:
            # - The train state's model parameters
            # - The optimizer state
            # - The training loss and accuracy batch metrics
            train_step(model, optimizer, metrics, batch)

            if step > 0 and (
                step % eval_every == 0
            ):  # One training epoch has passed.
                # Log the training metrics.
                for metric, value in metrics.compute().items():  # Compute the metrics.
                    mlflow.log_metric(f"train_{metric}", value)  # Record the metrics.
                metrics.reset()  # Reset the metrics for the test set.

                # # Compute the metrics on the test set after each training epoch.
                # for test_batch in test_ds.as_numpy_iterator():
                #     eval_step(model, metrics, test_batch)

                # # Log the test metrics.
                # for metric, value in metrics.compute().items():
                #     metrics_history[f"test_{metric}"].append(value)
                # metrics.reset()  # Reset the metrics for the next training epoch.
