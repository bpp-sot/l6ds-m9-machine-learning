import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs('docs/assets/images', exist_ok=True)
sns.set_theme(style='whitegrid')

# ---------------------------------------------------------
# 1. Linear Regression
# ---------------------------------------------------------
print("Generating plots for Linear Regression...")
df_tips = sns.load_dataset('tips')
X_lin = df_tips[['total_bill']]
y_lin = df_tips['tip']

lr = LinearRegression()
lr.fit(X_lin, y_lin)
preds_lin = lr.predict(X_lin)

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_tips, x='total_bill', y='tip', alpha=0.6, color='#2D2D2D')
plt.plot(df_tips['total_bill'], preds_lin, color='#D94D26', linewidth=2, label='Line of Best Fit')
plt.title('Linear Regression: Ordinary Least Squares')
plt.legend()
plt.tight_layout()
plt.savefig('docs/assets/images/topic3-linear-regression.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 2. Logistic Regression (Sigmoid / Decision Boundary)
# ---------------------------------------------------------
print("Generating plots for Logistic Regression...")
# Use diamonds for a binary classification task
df_diamonds = sns.load_dataset('diamonds').sample(1000, random_state=42)
df_diamonds['is_premium'] = np.where(df_diamonds['cut'] == 'Premium', 1, 0)

X_log = df_diamonds[['carat']]
y_log = df_diamonds['is_premium']

logreg = LogisticRegression()
logreg.fit(X_log, y_log)

X_test_log = np.linspace(0, 3, 300).reshape(-1, 1)
y_prob = logreg.predict_proba(X_test_log)[:, 1]

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_diamonds, x='carat', y='is_premium', alpha=0.3, color='#6E368A')
plt.plot(X_test_log, y_prob, color='#D94D26', linewidth=3, label='Sigmoid Probability Curve')
plt.axhline(0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Logistic Regression Sigmoid Activation')
plt.legend()
plt.tight_layout()
plt.savefig('docs/assets/images/topic3-logistic-regression.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 3. Decision Trees Contour
# ---------------------------------------------------------
print("Generating plots for Decision Trees...")
X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_moons, y_moons)

xx, yy = np.meshgrid(np.linspace(X_moons[:, 0].min()-0.5, X_moons[:, 0].max()+0.5, 100),
                     np.linspace(X_moons[:, 1].min()-0.5, X_moons[:, 1].max()+0.5, 100))
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')
sns.scatterplot(x=X_moons[:, 0], y=X_moons[:, 1], hue=y_moons, palette='Set2', edgecolor='k')
plt.title('Decision Tree Orthogonal Geometry (Depth 3)')
plt.tight_layout()
plt.savefig('docs/assets/images/topic3-decision-trees.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 4. Support Vector Machines (Margins)
# ---------------------------------------------------------
print("Generating plots for SVM...")
X_blobs, y_blobs = make_blobs(n_samples=100, centers=2, random_state=6)
svm = SVC(kernel='linear', C=1000)
svm.fit(X_blobs, y_blobs)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_blobs[:, 0], y=X_blobs[:, 1], hue=y_blobs, palette='Set1', s=50)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# plot decision boundary and margins
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title('SVM Linear Hyperplane and Margins')
plt.tight_layout()
plt.savefig('docs/assets/images/topic3-svm.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 5. Model Comparison (Confusion Matrices)
# ---------------------------------------------------------
print("Generating plots for Model Comparison...")
df_iris = sns.load_dataset('iris')
X_iris = df_iris.drop(columns='species')
y_iris = df_iris['species']
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

rf2 = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf2.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=rf2.classes_, yticklabels=rf2.classes_)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('docs/assets/images/topic3-model-comparison.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 6. Gradient Boosting (Learning Curve approximation)
# ---------------------------------------------------------
print("Generating plots for Gradient Boosting...")
# Train deviance (loss) over stages
X_moons, y_moons = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, random_state=42)

gbc = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)
gbc.fit(X_train, y_train)

train_loss = np.zeros((100,), dtype=np.float64)
test_loss = np.zeros((100,), dtype=np.float64)

for i, y_pred in enumerate(gbc.staged_predict_proba(X_train)):
    # Very simple proxy for loss tracking visually
    train_loss[i] = 1 - accuracy_score(y_train, np.argmax(y_pred, axis=1))
    
for i, y_pred in enumerate(gbc.staged_predict_proba(X_test)):
    test_loss[i] = 1 - accuracy_score(y_test, np.argmax(y_pred, axis=1))

plt.figure(figsize=(8, 5))
plt.plot(np.arange(100) + 1, train_loss, 'b-', label='Training Error')
plt.plot(np.arange(100) + 1, test_loss, 'r-', label='Validation Error')
plt.title('Gradient Boosting Convergence (Trees vs Error)')
plt.xlabel('Boosting Iterations (Trees)')
plt.ylabel('Misclassification Error')
plt.legend()
plt.tight_layout()
plt.savefig('docs/assets/images/topic3-gradient-boosting.png', dpi=150, facecolor='white')
plt.close()

print("All Topic 3 plots generated successfully.")
