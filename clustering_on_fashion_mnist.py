
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances

LABEL_MAP = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}
import numpy as np

def load_fashion_mnist_data(num_samples=None):
    """
    Loads Fashion-MNIST using Keras (robust against SSL errors).

    Args:
        num_samples (int, optional): If provided, returns a random subset of this size.
                                     If None, returns the full dataset (70,000 images).

    Returns:
        X (np.array): Flattened feature matrix of shape (N, 784).
        y (np.array): Label vector of shape (N,).
    """
    print("Loading Fashion-MNIST via TensorFlow/Keras...")


    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    X = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])


    X = X.reshape(X.shape[0], -1)

    if num_samples is not None and num_samples < X.shape[0]:
        print(f"   > Subsetting to {num_samples} random samples...")
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], num_samples, replace=False)
        X = X[indices]
        y = y[indices]

    X = X.astype(np.float32)

    print(f"   > Data Loaded. Shape: {X.shape}")
    return X, y


class PCA_FromScratch:
    def __init__(self, n_components, solver='eigen'):
        self.n_components = n_components
        self.solver = solver
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit the model with X.

        Tasks:
        1. Compute the Covariance Matrix of X.
        2. Perform Eigendecomposition.
        3. Sort eigenvectors by eigenvalues in descending order.
        4. Store the top 'n_components' eigenvectors.
        """
        n_samples, n_features = X.shape

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_



        if self.solver == 'eigen':
          cov_matrix= X_centered.T.dot(X_centered) / (n_samples - 1)
          eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
          idx = np.argsort(eigenvalues)[::-1]
          eigenvalues = eigenvalues[idx]
          eigenvectors = eigenvectors[:, idx]

          self.components_ = eigenvectors[:, :self.n_components].T
          self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

        elif self.solver == 'svd':
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components = Vt[:, :self.n_components]
            eigenvalues_from_svd = (S ** 2) / (n_samples - 1)
            self.explained_variance_ratio_ = (
                eigenvalues_from_svd[:self.n_components] / np.sum(eigenvalues_from_svd)
            )

        pass

    def transform(self, X):
        """
        Project X onto the principal components.
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
        pass

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the original data from the projected data.
        """
        return np.dot(X_transformed, self.components_) + self.mean_
        pass

def plot_projection(X_proj, y, title):
    """Plots the 2D projection of the data."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
    plt.legend(handles=scatter.legend_elements()[0],
               labels=[LABEL_MAP[i] for i in range(10)],
               title="Classes")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_eigen_clothing(components):
    """Visualizes the top components as images."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Top 10 Eigen-Clothing (Principal Components)")
    for i, ax in enumerate(axes.flat):

        img = components[i].reshape(28, 28)
        ax.imshow(img, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"PC {i+1}")
    plt.show()

if __name__ == "__main__":

    X, y = load_fashion_mnist_data(num_samples=2000)
    print("\n--- Running PCA From Scratch ---")
    pca = PCA_FromScratch(n_components=20)
    pca.fit(X)

    if pca.components_ is None:
        print("ERR: PCA fit method not implemented yet.")
    else:

        X_pca_2d = pca.transform(X)[:, :2]
        plot_projection(X_pca_2d, y, "PCA Projection (Unsupervised)")

        plot_eigen_clothing(pca.components_)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance Ratio')
        plt.grid()
        plt.show()
    print("\n--- Running LDA (sklearn) ---")
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)

    plot_projection(X_lda, y, "LDA Projection (Supervised)")

    print("Done.")


def run_tsne_experiment(X, y, perplexities=[5, 30, 100]):
    print(f"\n--- Running t-SNE Experiments ---")


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"t-SNE: Impact of Perplexity")

    for i, perp in enumerate(perplexities):
        print(f"Running t-SNE with Perplexity = {perp}...")

        tsne =  TSNE(n_components=2, perplexity=perp)
        X_tsne = tsne.fit_transform(X)


        ax = axes[i]
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
        ax.set_title(f"Perplexity: {perp}")
        ax.grid(True, alpha=0.3)


        ax.set_xticks([])
        ax.set_yticks([])


    handles, _ = scatter.legend_elements()
    labels = [LABEL_MAP[i] for i in range(10)]
    fig.legend(handles, labels, loc='center right', title="Classes")
    plt.tight_layout()
    plt.subplots_adjust(right=0.85) 
    plt.show()

if __name__ == "__main__":
    X, y = load_fashion_mnist_data(num_samples=1000)


    run_tsne_experiment(X, y)


