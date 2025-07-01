# 📊 LOGIQUE DE CALCUL DES PROMOTIONS - GUIDE DÉTAILLÉ

## 🎯 Vue d'ensemble du système

Le système d'IA pour les promotions utilise une approche multicritères basée sur :

1. **Calcul des KPIs de performance commerciale**
2. **Système de scoring des règles métier**
3. **Modèles d'apprentissage automatique**
4. **Évaluation des risques et ROI**

---

## 📈 1. CALCUL DES KPIs (Indicateurs de Performance)

### 1.1 Taux de Rotation (Rotation Rate)

**Formule :** `Rotation = Ventes totales 90j / Quantité injectée`

```python
def calculate_rotation(total_sales, quantity_injected):
    if quantity_injected == 0:
        return 0.0
    return total_sales / quantity_injected
```

**Exemple concret :**

- Produit : Robe été 2024
- Ventes 90 jours : 45 unités
- Quantité injectée : 100 unités
- **Rotation = 45/100 = 0.45 (45%)**

**Interprétation :**

- Rotation < 0.3 → Produit lent (candidat à promotion)
- Rotation 0.3-0.7 → Performance moyenne
- Rotation > 0.7 → Produit performant

### 1.2 Taux de Vente (Sell-Through Rate)

**Formule :** `Taux vente = Unités vendues / Total disponible`

```python
def calculate_sell_through_rate(units_sold, total_available):
    if total_available == 0:
        return 0.0
    return units_sold / total_available
```

**Exemple concret :**

- Produit : Jean slim noir
- Unités vendues : 35
- Stock initial + livraisons : 120 unités
- **Taux de vente = 35/120 = 0.29 (29%)**

### 1.3 Couverture de Stock (en jours)

**Formule :** `Couverture = Stock actuel / Vente moyenne quotidienne`

```python
def calculate_stock_coverage_days(current_stock, daily_sales_rate):
    if daily_sales_rate == 0:
        return 999  # Couverture infinie si pas de ventes
    return current_stock / daily_sales_rate
```

**Exemple concret :**

- Produit : Veste blazer
- Stock actuel : 25 unités
- Ventes quotidiennes moyennes : 0.5 unités/jour
- **Couverture = 25/0.5 = 50 jours**

### 1.4 Rotation d'Inventaire

**Formule :** `Rotation inventaire = Ventes annuelles / Stock moyen`

### 1.5 Marge Brute

**Formule :** `Marge = (Prix vente - Prix achat) / Prix vente`

---

## 🎯 2. SYSTÈME DE SCORING POUR LA DÉCISION PROMOTIONNELLE

### 2.1 Règles de scoring (create_promotion_labels)

Le système attribue des points selon 6 critères principaux :

#### **Règle 1 : Stock élevé + Faibles ventes**

```python
# Points attribués selon stock et ventes
if current_stock > 50 and sales_90d < 5:
    promotion_score += 3  # Situation critique
elif current_stock > 20 and sales_90d < 10:
    promotion_score += 2  # Situation préoccupante
elif current_stock > 10 and sales_90d < 15:
    promotion_score += 1  # Attention requise
```

**Exemple :**

- Produit A : Stock = 60, Ventes 90j = 3 → **+3 points**
- Produit B : Stock = 25, Ventes 90j = 8 → **+2 points**

#### **Règle 2 : Tendance des ventes**

```python
# Évaluation de la tendance
if sales_trend < -0.3:        # Chute > 30%
    promotion_score += 2
elif sales_trend < -0.1:      # Chute > 10%
    promotion_score += 1
```

**Exemple :**

- Ventes mois dernier : 20 unités
- Ventes mois précédent : 30 unités
- Tendance = (20-30)/30 = -0.33 (-33%) → **+2 points**

#### **Règle 3 : Temps depuis dernière promotion**

```python
if days_since_promo > 180:    # 6 mois
    promotion_score += 2
elif days_since_promo > 120:  # 4 mois
    promotion_score += 1
```

#### **Règle 4 : Taux de rotation faible**

```python
if rotation < 0.1:
    promotion_score += 3      # Rotation critique
elif rotation < 0.3:
    promotion_score += 2      # Rotation faible
elif rotation < 0.5:
    promotion_score += 1      # Rotation moyenne
```

#### **Règle 5 : Couverture de stock excessive**

```python
if stock_coverage > 365:     # Plus d'un an
    promotion_score += 3
elif stock_coverage > 180:   # 6 mois
    promotion_score += 2
elif stock_coverage > 90:    # 3 mois
    promotion_score += 1
```

#### **Règle 6 : Faible taux de vente**

```python
if sell_through < 0.1:       # Moins de 10%
    promotion_score += 2
elif sell_through < 0.3:     # Moins de 30%
    promotion_score += 1
```

### 2.2 Décision finale

```python
# Décision basée sur le score total
should_promote = promotion_score >= 4  # Seuil de 4 points

# Conditions d'exclusion
if days_since_promo < 30:               # Promotion récente
    should_promote = False

if profit_margin < 0.15:                # Marge insuffisante
    should_promote = False
```

---

## 🤖 3. MODÈLES D'APPRENTISSAGE AUTOMATIQUE

### 3.1 Classificateur de Promotion (RandomForestClassifier)

**Objectif :** Prédire si un produit doit être en promotion

**Features utilisées :**

```python
features = [
    "current_price",         # Prix actuel
    "current_stock",         # Stock actuel
    "total_sales_90d",       # Ventes 90 jours
    "rotation",              # Taux de rotation
    "sell_through_rate",     # Taux de vente
    "stock_coverage_days",   # Couverture en jours
    "inventory_turnover",    # Rotation d'inventaire
    "sales_trend",           # Tendance des ventes
    "profit_margin",         # Marge bénéficiaire
    "days_since_last_promo", # Jours depuis dernière promo
    "last_promo_discount",   # Dernier taux de réduction
    "promo_count_6months"    # Nombre de promos 6 mois
]
```

### 3.2 Régresseur de Taux de Réduction (RandomForestRegressor)

**Objectif :** Calculer le taux de réduction optimal

**Logique de calcul :**

```python
def _generate_synthetic_discount(row):
    base_discount = 0.15  # Base 15%

    # Ajustement selon couverture stock
    if stock_coverage > 180:
        adjustment += 0.1    # +10% si surstock
    elif stock_coverage > 90:
        adjustment += 0.05   # +5% si stock élevé

    # Ajustement selon rotation
    if rotation < 0.2:
        adjustment += 0.05   # +5% si rotation faible

    return min(0.3, max(0.1, base_discount + adjustment))
```

### 3.3 Régresseur d'Impact (RandomForestRegressor)

**Objectif :** Prédire l'augmentation des ventes

**Facteurs d'impact :**

```python
def _generate_synthetic_impact(row):
    base_lift = discount * 2.0  # Base : 2x le taux de réduction

    # Facteur prix (produits chers répondent mieux)
    if price > 150:
        price_factor = 1.2
    elif price > 100:
        price_factor = 1.1
    else:
        price_factor = 1.0

    # Facteur performance (produits lents répondent mieux)
    if rotation < 0.2:
        performance_factor = 1.3
    elif sell_through < 0.3:
        performance_factor = 1.2
    else:
        performance_factor = 1.0

    return base_lift * price_factor * performance_factor
```

---

## 📊 4. EXEMPLES CONCRETS DE CALCUL

### Exemple 1 : Produit nécessitant une promotion urgente

**Données produit :**

- Nom : "Robe été fleurie"
- Prix : 89.99 €
- Stock actuel : 55 unités
- Ventes 90j : 4 unités
- Ventes mois dernier : 1 unité
- Ventes mois précédent : 3 unités
- Dernière promotion : il y a 8 mois

**Calcul des KPIs :**

```
Rotation = 4 / (4 + 55) = 0.068 (6.8%)
Taux de vente = 4 / 59 = 0.068 (6.8%)
Couverture stock = 55 / (4/90) = 1,237 jours
Tendance = (1-3)/3 = -67%
```

**Calcul du score de promotion :**

```
Règle 1: Stock=55, Ventes=4  → +3 points (critique)
Règle 2: Tendance=-67%       → +2 points (chute forte)
Règle 3: 8 mois sans promo   → +2 points
Règle 4: Rotation=0.068      → +3 points (très faible)
Règle 5: Couverture=1237j    → +3 points (excessive)
Règle 6: Taux vente=6.8%     → +2 points (très faible)

SCORE TOTAL = 15 points (≥ 4) → PROMOTION RECOMMANDÉE
```

**Prédiction IA :**

```
Discount optimal : 25% (base 15% + 10% surstock)
Impact prévu : +120% ventes (réduction 25% × 2 × facteur produit lent 1.3 × facteur prix moyen 1.1)
Revenus attendus : 89.99€ × 1 unité/mois × 2.2 = +108€/mois
Confiance : 95%
```

### Exemple 2 : Produit performant (pas de promotion)

**Données produit :**

- Nom : "T-shirt basique blanc"
- Prix : 24.99 €
- Stock actuel : 15 unités
- Ventes 90j : 45 unités
- Tendance : +10%
- Dernière promotion : il y a 2 mois

**Calcul des KPIs :**

```
Rotation = 45 / (45 + 15) = 0.75 (75%)
Taux de vente = 45 / 60 = 0.75 (75%)
Couverture stock = 15 / (45/90) = 30 jours
```

**Score de promotion :**

```
Toutes les règles → 0 points (performance satisfaisante)
SCORE TOTAL = 0 points (< 4) → PAS DE PROMOTION
```

---

## 🎯 5. CALCUL DE LA DURÉE DE PROMOTION

### Algorithme de calcul

```python
def predict_promotion_end_date(start_date, category_data, optimal_discount):
    base_duration = 14  # Base 2 semaines

    # Ajustements selon discount
    if optimal_discount > 0.25:
        duration += 7    # +1 semaine si forte réduction
    elif optimal_discount > 0.15:
        duration += 3    # +3 jours si réduction modérée

    # Ajustements selon stock
    if avg_stock_coverage > 90:
        duration += 10   # +10 jours si surstock

    # Ajustements selon rotation
    if avg_rotation < 0.2:
        duration += 7    # +1 semaine si rotation faible

    # Ajustements selon prix (produits premium)
    if avg_price > 200:
        duration = max(7, duration - 5)  # Durée plus courte

    return max(7, min(28, duration))  # Entre 1 et 4 semaines
```

---

## 💰 6. CALCUL DU ROI ET IMPACT FINANCIER

### Revenus attendus

```python
monthly_baseline = price * (sales_90d / 3)  # Revenus mensuels de base
revenue_increase = baseline * predicted_sales_lift
net_revenue = revenue_increase - (price * discount * new_sales)
```

### ROI de la promotion

```python
promotion_cost = price * discount * units_sold_during_promo
revenue_gain = additional_revenue_from_lift
roi = (revenue_gain - promotion_cost) / promotion_cost * 100
```

### Exemple de calcul ROI

**Produit : Veste 120€**

- Discount : 20%
- Ventes baseline : 5 unités/mois
- Impact prévu : +80%
- Ventes promo : 5 × 1.8 = 9 unités

```
Coût promotion = 120€ × 0.20 × 9 = 216€
Revenus supplémentaires = 120€ × (9-5) = 480€
ROI = (480€ - 216€) / 216€ × 100 = 122%
```

---

## 🚨 7. ÉVALUATION DES RISQUES

### Niveaux de risque

```python
if profit_margin < 0.2:
    risk = "HIGH - Low profit margin"
elif optimal_discount > 0.25:
    risk = "MEDIUM - High discount"
elif days_since_last_promo < 30:
    risk = "MEDIUM - Recent promotion"
else:
    risk = "LOW"
```

### Facteurs de risque considérés

1. **Marge bénéficiaire** : Risque de perte si marge < 20%
2. **Taux de réduction** : Risque d'impact sur image si > 25%
3. **Fréquence promotions** : Risque de cannibalisation
4. **Saisonnalité** : Risque de timing inapproprié
5. **Concurrence** : Risque de guerre des prix

---

## 🎯 8. WORKFLOW COMPLET - EXEMPLE PRATIQUE

### Étape 1 : Extraction des données

```sql
-- Données produit avec ventes et stock
SELECT a.*, s.QuantitePhysique, v.sales_data
FROM Articles a
JOIN Stocks s ON a.Id = s.ArticleId
JOIN (SELECT SUM(sales) as sales_data FROM Ventes) v
```

### Étape 2 : Calcul des KPIs

```python
df["rotation"] = calculate_rotation(sales_90d, total_purchased)
df["sell_through"] = calculate_sell_through_rate(sales, available)
df["stock_coverage"] = calculate_stock_coverage_days(stock, daily_sales)
```

### Étape 3 : Scoring et décision

```python
promotion_score = evaluate_business_rules(product_data)
should_promote = promotion_score >= 4 and check_exclusions()
```

### Étape 4 : Prédiction IA

```python
optimal_discount = discount_regressor.predict(features)
sales_lift = impact_regressor.predict(features)
confidence = promotion_classifier.predict_proba(features)
```

### Étape 5 : Calcul financier

```python
expected_revenue = baseline_revenue * sales_lift
promotion_cost = price * discount * projected_sales
roi = (expected_revenue - promotion_cost) / promotion_cost
```

---

## 📋 9. RÉSUMÉ DES SEUILS CRITIQUES

| Métrique         | Seuil Critique | Action                |
| ---------------- | -------------- | --------------------- |
| Rotation         | < 0.3          | Promotion recommandée |
| Taux de vente    | < 30%          | Surveillance accrue   |
| Couverture stock | > 90 jours     | Action immédiate      |
| Tendance ventes  | < -20%         | Promotion urgente     |
| Marge profit     | < 15%          | Exclusion promotion   |
| Dernière promo   | < 30 jours     | Attendre              |

Cette logique garantit des décisions promotionnelles basées sur des données factuelles tout en préservant la rentabilité et l'image de marque.
