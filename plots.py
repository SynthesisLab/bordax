import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def process_metrics(metrics_list, num_checkpoints, num_epochs):
    """
    Process metrics from training into visualizable format.

    Args:
        metrics_list: List of dicts, each with keys 'value_loss', 'entropy_loss', 'loss'
                     Values have shape (num_sgd_steps, num_minibatches)
        num_checkpoints: Number of checkpoints
        num_epochs: Number of epochs per checkpoint

    Returns:
        Dict with processed metrics for visualization
    """
    # Extract each metric type
    loss_keys = ["loss", "value_loss", "entropy_loss", "total_loss"]
    processed = {key: [] for key in loss_keys}

    for metrics_dict in metrics_list:
        for key in loss_keys:
            if key in metrics_dict:
                # Shape: (num_sgd_steps, num_minibatches)
                metric_array = metrics_dict[key]

                # Average over minibatches for each SGD step
                # Result shape: (num_sgd_steps,)
                sgd_averages = jnp.mean(metric_array, axis=1)
                processed[key].append(sgd_averages)

    # Convert to arrays
    # Shape: (num_checkpoints * num_epochs, num_sgd_steps)
    for key in loss_keys:
        processed[key] = np.array(processed[key])

    return processed


def visualize_training_metrics(
    metrics_list, num_checkpoints, num_epochs, save_path=None, figsize=(10, 6)
):
    """
    Create comprehensive visualization of training metrics.
    """
    processed = process_metrics(metrics_list, num_checkpoints, num_epochs)

    num_sgd_steps = processed["loss"].shape[1]
    total_updates = num_checkpoints * num_epochs

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Loss evolution across all training
    ax = axes[0]
    loss_flat = processed["total_loss"].flatten()
    ax.plot(loss_flat, alpha=0.7)
    ax.set_title("Total Loss Evolution")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Loss")

    # # Add epoch boundaries
    # for i in range(0, total_updates, num_epochs):
    #     ax.axvline(x=i * num_sgd_steps, color="gray", linestyle="--", alpha=0.3)

    # # 2. SGD steps within epochs (show pattern)
    # ax = axes[0, 1]
    # # Show first few epochs to see SGD pattern
    # epochs_to_show = min(5, total_updates)
    # for epoch in range(epochs_to_show):
    #     sgd_losses = processed["loss"][epoch]
    #     ax.plot(
    #         range(num_sgd_steps),
    #         sgd_losses,
    #         marker="o",
    #         label=f"Epoch {epoch}",
    #         alpha=0.7,
    #     )
    # ax.set_title("Loss per SGD Step (First Few Epochs)")
    # ax.set_xlabel("SGD Step")
    # ax.set_ylabel("Loss")
    # ax.legend()

    # # 3. Average loss per epoch
    # ax = axes[0, 2]
    # epoch_avg_loss = np.mean(processed["loss"], axis=1)
    # ax.plot(epoch_avg_loss)
    # ax.set_title("Average Loss per Epoch")
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Average Loss")

    # # 4. Value loss evolution
    # ax = axes[1, 0]
    # value_loss_flat = processed["value_loss"].flatten()
    # ax.plot(value_loss_flat, alpha=0.7, color="orange")
    # ax.set_title("Value Loss Evolution")
    # ax.set_xlabel("Update Step")
    # ax.set_ylabel("Value Loss")

    # # 5. Entropy loss evolution
    # ax = axes[1, 1]
    # entropy_loss_flat = processed["entropy_loss"].flatten()
    # ax.plot(entropy_loss_flat, alpha=0.7, color="green")
    # ax.set_title("Entropy Loss Evolution")
    # ax.set_xlabel("Update Step")
    # ax.set_ylabel("Entropy Loss")

    # 6. All losses comparison (normalized)
    ax = axes[1]
    # Normalize each loss type for comparison
    for key, color in [
        ("loss", "blue"),
        ("value_loss", "orange"),
        ("entropy_loss", "green"),
    ]:
        data = jnp.mean(processed[key], axis=1)  # Average per epoch
        # Normalize to [0, 1] for comparison
        # data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        ax.plot(data, label=key, color=color, alpha=0.7)
    ax.set_title("Losses Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Loss")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

    return processed
