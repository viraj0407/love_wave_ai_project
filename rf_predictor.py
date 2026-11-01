"""
Random Forest Predictor & Feature Importance Analyzer for Love Waves in Smart Composites
Provides forward AI prediction of phase velocity and ranks the physical variables that matter most.
"""

import numpy as np
import pandas as pd

class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None

class PureDecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=4, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.max_features_ = self.max_features or self.n_features_
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) < 1e-6:
            return DecisionNode(value=np.mean(y))

        # Sample features
        feat_indices = np.random.choice(n_features, self.max_features_, replace=False)
        
        best_feat, best_thresh = None, None
        best_var_reduction = -1.0
        current_var = np.var(y) * n_samples

        for f_idx in feat_indices:
            thresholds = np.percentile(X[:, f_idx], np.linspace(10, 90, 9))
            for thresh in thresholds:
                left_mask = X[:, f_idx] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                var_left = np.var(y[left_mask]) * np.sum(left_mask)
                var_right = np.var(y[right_mask]) * np.sum(right_mask)
                var_red = current_var - (var_left + var_right)
                if var_red > best_var_reduction:
                    best_var_reduction = var_red
                    best_feat = f_idx
                    best_thresh = thresh

        if best_var_reduction <= 0 or best_feat is None:
            return DecisionNode(value=np.mean(y))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(feature_idx=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

class PureRandomForestRegressor:
    def __init__(self, n_estimators=35, max_depth=12, min_samples_split=4, max_features=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.max_features_ = self.max_features or max(1, int(np.sqrt(n_features) * 1.5))
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            tree = PureDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features_
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

def train_and_evaluate(df):
    feature_cols = ['kh1', 'thickness_um', 'micro_inertia_nm', 'flexo_coeff_nC_m', 
                    'piezo_coupling', 'mechanical_slip_RM', 'electrical_leak_RE']
    target_col = 'phase_velocity_mps'

    X = df[feature_cols].values
    y = df[target_col].values

    # Train-test split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Training Random Forest model on {len(X_train)} samples, testing on {len(X_test)} samples...")
    rf = PureRandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # Metrics
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    ss_res = np.sum((y_test - y_pred)**2)
    r2 = 1.0 - (ss_res / ss_tot)

    print(f"--- Model Performance Evaluation ---")
    print(f"Test R^2 Score: {r2:.4f}")
    print(f"Test RMSE:      {rmse:.2f} m/s")
    print(f"Test MAE:       {mae:.2f} m/s")

    # Permutation Feature Importance
    print("\nComputing Permutation Feature Importance ranking...")
    baseline_mse = np.mean((y_test - y_pred)**2)
    importances = {}

    for i, col in enumerate(feature_cols):
        X_perm = X_test.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        y_perm_pred = rf.predict(X_perm)
        perm_mse = np.mean((y_test - y_perm_pred)**2)
        importances[col] = max(0.0, perm_mse - baseline_mse)

    total_imp = sum(importances.values()) or 1.0
    normalized_imp = {k: v / total_imp for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

    print("--- Physical Variables Importance Ranking ---")
    for rank, (feat, score) in enumerate(normalized_imp.items(), 1):
        print(f"{rank}. {feat:<20}: {score*100:6.2f}%")

    return rf, normalized_imp, (r2, rmse, mae)

if __name__ == "__main__":
    df = pd.read_csv("artifacts/file_generation/ttl=63d/intermediate/love_wave_ai_project/love_wave_dataset.csv")
    rf, imp, metrics = train_and_evaluate(df)
