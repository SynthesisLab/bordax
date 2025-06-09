import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle
import gzip

from planning import RolloutWithActivations

# def create_trajectory_prediction_dataset()

def create_trajectory_prediction_dataset(
    rollouts: List[RolloutWithActivations],
    n_negative_per_positive: int = 3,
    spatial_resolution: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dataset for trajectory prediction.
    Each sample is [initial_observation, x, y] -> {0, 1}
    """
    positive_samples = []
    negative_samples = []

    all_points = []
    for rollout in rollouts:
        all_points.extend(rollout.observations[:, :2])
    all_points = np.array(all_points)

    # Get spatial bounds
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    print(f"Spatial bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")

    for rollout in rollouts:
        initial_obs = rollout.observations[0]
        trajectory_xy = rollout.observations[:, :2]

        # for x, y in trajectory_xy:
        #     sample = np.concatenate([initial_obs, [x, y]])
        #     positive_samples.append(sample)

        # Discretize trajectory to avoid too many similar points
        visited_cells = set()
        for x, y in trajectory_xy:
            cell_x = int(x / spatial_resolution)
            cell_y = int(y / spatial_resolution)
            visited_cells.add((cell_x, cell_y))

        # Positive samples: points on the trajectory
        for cell_x, cell_y in visited_cells:
            x = cell_x * spatial_resolution
            y = cell_y * spatial_resolution
            # Combine initial observation with query point
            sample = np.concatenate([initial_obs, [x, y]])
            positive_samples.append(sample)

        # Negative samples: random points not on trajectory
        n_negatives = len(visited_cells) * n_negative_per_positive
        for _ in range(n_negatives):
            # Sample random point in the space
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            # Check if it's far enough from trajectory
            distances = np.sqrt(
                (trajectory_xy[:, 0] - x) ** 2 + (trajectory_xy[:, 1] - y) ** 2
            )

            if np.min(distances) > spatial_resolution * 2:
                sample = np.concatenate([initial_obs, [x, y]])
                negative_samples.append(sample)

    # Combine positive and negative samples
    X_pos = np.array(positive_samples)
    X_neg = np.array(negative_samples)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    print(f"\nDataset created:")
    print(f"  Positive samples: {len(X_pos)} ({len(X_pos)/len(X)*100:.1f}%)")
    print(f"  Negative samples: {len(X_neg)} ({len(X_neg)/len(X)*100:.1f}%)")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimension: {X.shape[1]} (initial_obs + x + y)")

    return X, y


def train_trajectory_predictor(
    rollouts: List[RolloutWithActivations],
    test_size: float = 0.3,
    model_type: str = "mlp",
) -> Pipeline:
    """Train a model to predict trajectory occupancy."""

    # Create dataset
    X, y = create_trajectory_prediction_dataset(rollouts)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Choose model
    if model_type == "logistic":
        classifier = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    elif model_type == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(10),
            activation="relu",
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
        )

    # Create pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    # Train
    print(f"\nTraining {model_type} model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nResults:")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline


def predict_trajectory_heatmap(
    pipeline: Pipeline,
    initial_obs: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 100,
) -> np.ndarray:
    """Generate a heatmap of trajectory probability given initial observation."""

    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)

    heatmap = np.zeros((resolution, resolution))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            # Create query: [initial_obs, x, y]
            query = np.concatenate([initial_obs, [x, y]]).reshape(1, -1)

            # Get probability
            prob = pipeline.predict_proba(query)[0, 1]
            heatmap[i, j] = prob

    return heatmap, x_vals, y_vals


def visualize_trajectory_predictions(
    rollouts: List[RolloutWithActivations], pipeline: Pipeline, n_examples: int = 4
):
    """Visualize predicted trajectories vs actual."""

    fig, axes = plt.subplots(2, n_examples, figsize=(5 * n_examples, 10))

    # Sample random rollouts
    indices = np.random.choice(len(rollouts), n_examples, replace=False)

    for idx, rollout_idx in enumerate(indices):
        rollout = rollouts[rollout_idx]
        initial_obs = rollout.observations[0]
        actual_trajectory = rollout.observations[:, :2]

        # Get spatial bounds
        x_min, x_max = (
            actual_trajectory[:, 0].min() - 1,
            actual_trajectory[:, 0].max() + 1,
        )
        y_min, y_max = (
            actual_trajectory[:, 1].min() - 1,
            actual_trajectory[:, 1].max() + 1,
        )

        # Generate heatmap
        heatmap, x_vals, y_vals = predict_trajectory_heatmap(
            pipeline, initial_obs, (x_min, x_max), (y_min, y_max)
        )

        # Plot heatmap
        ax = axes[0, idx]
        im = ax.imshow(
            heatmap,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="hot",
            aspect="auto",
        )
        ax.plot(
            actual_trajectory[:, 0],
            actual_trajectory[:, 1],
            "b-",
            linewidth=2,
            label="Actual trajectory",
        )
        ax.scatter(
            actual_trajectory[0, 0],
            actual_trajectory[0, 1],
            color="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax.scatter(
            actual_trajectory[-1, 0],
            actual_trajectory[-1, 1],
            color="red",
            s=100,
            marker="x",
            label="End",
        )
        ax.set_title(f"Predicted Occupancy (Episode {rollout_idx})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.colorbar(im, ax=ax)

        # Plot thresholded prediction
        ax = axes[1, idx]
        threshold = 0.5
        binary_pred = (heatmap > threshold).astype(float)
        ax.imshow(
            binary_pred,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="RdBu",
            aspect="auto",
        )
        ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], "g-", linewidth=2)
        ax.set_title(f"Binary Prediction (threshold={threshold})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()


def analyze_prediction_quality(
    rollouts: List[RolloutWithActivations], pipeline: Pipeline, n_samples: int = 100
):
    """Analyze how well the model predicts trajectories."""

    metrics = []

    for i in range(min(n_samples, len(rollouts))):
        rollout = rollouts[i]
        initial_obs = rollout.observations[0]
        trajectory = rollout.observations[:, :2]

        # Sample points along trajectory
        true_positives = 0
        for x, y in trajectory[::5]:  # Every 5th point
            query = np.concatenate([initial_obs, [x, y]]).reshape(1, -1)
            prob = pipeline.predict_proba(query)[0, 1]
            if prob > 0.5:
                true_positives += 1

        # Sample points not on trajectory
        false_positives = 0
        for _ in range(len(trajectory) // 5):
            # Random point in space
            x = np.random.uniform(
                trajectory[:, 0].min() - 2, trajectory[:, 0].max() + 2
            )
            y = np.random.uniform(
                trajectory[:, 1].min() - 2, trajectory[:, 1].max() + 2
            )

            # Check if actually on trajectory
            distances = np.sqrt(
                (trajectory[:, 0] - x) ** 2 + (trajectory[:, 1] - y) ** 2
            )
            if np.min(distances) > 0.1:  # Not on trajectory
                query = np.concatenate([initial_obs, [x, y]]).reshape(1, -1)
                prob = pipeline.predict_proba(query)[0, 1]
                if prob > 0.5:
                    false_positives += 1

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (len(trajectory) // 5)

        metrics.append(
            {
                "precision": precision,
                "recall": recall,
                "episode_return": rollout.episode_return,
            }
        )

    metrics = pd.DataFrame(metrics)

    print("\nTrajectory Prediction Quality:")
    print(f"Average Precision: {metrics['precision'].mean():.3f}")
    print(f"Average Recall: {metrics['recall'].mean():.3f}")

    # Correlation with episode performance
    corr = metrics[["precision", "recall", "episode_return"]].corr()
    print("\nCorrelation with episode return:")
    print(f"Precision: {corr.loc['precision', 'episode_return']:.3f}")
    print(f"Recall: {corr.loc['recall', 'episode_return']:.3f}")

    return metrics


# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Load rollouts
    with gzip.open("rollouts_with_activations.pkl.gz", "rb") as f:
        rollouts = pickle.load(f)

    # Train model
    pipeline = train_trajectory_predictor(rollouts, model_type="logistic")

    # Visualize predictions
    visualize_trajectory_predictions(rollouts, pipeline, n_examples=4)

    # Analyze quality
    metrics = analyze_prediction_quality(rollouts, pipeline)

    # Save model
    with open("trajectory_predictor.pkl", "wb") as f:
        pickle.dump(pipeline, f)
