# 📈 Ensemble Learning with Python

Repozytorium zawiera projekty i eksperymenty związane z metodami **Ensemble Learning** — technikami łączenia wielu modeli w celu poprawy jakości predykcji w zadaniach klasyfikacji i regresji.

Celem projektu jest praktyczne zrozumienie działania metod zespołowych oraz porównanie ich skuteczności względem pojedynczych modeli.

---

## 🚀 Czym jest Ensemble Learning?

Ensemble Learning polega na łączeniu wielu modeli bazowych w jeden silniejszy model predykcyjny.

Zamiast polegać na jednym algorytmie, wykorzystujemy wiele modeli, aby:

- zwiększyć dokładność predykcji  
- zmniejszyć wariancję (overfitting)  
- poprawić stabilność modelu  
- lepiej generalizować na nowe dane  

---

## 🧠 Zastosowane techniki

Projekt obejmuje różne podejścia do ensemble learning:

### 🔹 Bagging
- Random Forest  
- BaggingClassifier / BaggingRegressor  

Redukuje wariancję poprzez trenowanie wielu modeli na losowych próbkach danych.

---

### 🔹 Boosting
- AdaBoost  
- Gradient Boosting  
- (opcjonalnie) XGBoost / LightGBM  

Redukuje bias poprzez sekwencyjne trenowanie modeli, gdzie każdy kolejny poprawia błędy poprzedniego.

---

### 🔹 Stacking
- Łączenie wielu modeli bazowych  
- Model meta-learner (np. Logistic Regression)  

Pozwala wykorzystać mocne strony różnych algorytmów jednocześnie.

---

## 📁 Struktura repozytorium

```
├── datasets/                                # Zbiory danych używane w przykładach
├── 01_voting_classife.ipynb                  # Notebook: klasyfikacja głosowania
├── 02_baggin_and_pasting_clasification.ipynb  # Notebook: bagging i pasting (klasyfikacja)
├── 02_bagging_and_pasting_regression.ipynb   # Notebook: bagging i pasting (regresja)
├── ensembles.ipynb                           # Notebook: ogólne ensemble / porównania
├── pyproject.toml                            # Konfiguracja projektu / zależności
└── README.md                                 # Dokumentacja projektu
```

---



## ▶️ Jak uruchomić projekt

1. Sklonuj repozytorium:

```bash
git clone https://github.com/MateuszPietkiewicz/ensamble_learning.git
cd ensamble_learning
```

2. (Opcjonalnie) utwórz środowisko wirtualne:

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

3. Zainstaluj wymagane pakiety:

```bash
pip install -r requirements.txt
```

4. Uruchom Jupyter Notebook:

```bash
jupyter notebook
```

---

## 💻 Przykład — Stacking Classifier

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Dane przykładowe
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modele bazowe
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier())
]

# Model stacking
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack_model.fit(X_train, y_train)
pred = stack_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
```

---

## 📊 Cel projektu

- praktyczne zastosowanie metod ensemble learning  
- porównanie skuteczności różnych podejść 