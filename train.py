import pandas as pd
import numpy as np

data = pd.read_csv('dataset.csv')
data['forecast_date'] = pd.to_datetime(data['forecast_date'])

train_data = data[data['forecast_date'] < '2024-01-01']
test_data = data[data['forecast_date'] >= '2024-01-01']

X_train = train_data.drop('target', axis=1)
y_train = train_data['target']

X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

#Расчёт бейзлайна

coef_one = pd.read_csv('dataset/reference/coef_koof_one2.csv')
coef_four = pd.read_csv('dataset/reference/coef_koof_four2.csv')
coef_vol = pd.read_csv('dataset/reference/coef_koof_volume2.csv')
res_ref = pd.read_csv('dataset/assets/res_reference.csv')

test_eval = test_data.copy()
test_eval = test_eval.merge(
    res_ref[['res_id', 'filial', 'es', 'res', 'izmoroz_koef']],
    on='res_id',
    how='left'
)

test_eval['temp_day'] = test_eval.groupby('res_id')['temp_day'].transform(
    lambda x : x.fillna(method='ffill').fillna(method='bfill')
)

test_eval['wind_speed'] = test_eval.groupby('res_id')['wind_speed'].transform(
    lambda x : x.fillna(x.median())
)
test_eval['wind_speed'] = test_eval['wind_speed'].fillna(test_eval['wind_speed'].median())

test_eval['humidity'] = test_eval.groupby('res_id')['humidity'].transform(
    lambda x : x.fillna(x.median())
)
test_eval['humidity'] = test_eval['humidity'].fillna(test_eval['humidity'].median())

#разделение на периоды

is_summer = test_eval['forecast_date'].dt.month.isin([5, 6, 7, 8, 9])
summer_eval = test_eval[is_summer].copy()
winter_eval = test_eval[~is_summer].copy()
winter_eval = winter_eval.sort_values(['res_id', 'forecast_date'])

t = winter_eval['temp_day'].values
w = winter_eval['wind_speed'].values
h = winter_eval['humidity'].values
rain = winter_eval['rain'].fillna(0).values
snow = winter_eval['snow'].fillna(0).values
precip = rain + snow
dt = winter_eval['delta_t_24h'].fillna(0.0).values

#снегоналипание
koof_t_snow = np.select(
    [t < -15, t < -12, t < -8, t < -4, t < -2, t <= 2],
    [0.0, 1.1, 1.3, 2.0, 3.3, 5.5],
    default=0.0
)

koof_w_snow = np.select(
    [w < 2, w < 4, w < 8, w < 12],
    [1.4, 1.3, 1.2, 1.1],
    default=1.0
)

koof_o_snow = np.select(
    [precip < 2, precip < 4, precip < 10, precip < 15],
    [0.0, 1.1, 1.2, 1.3],
    default=1.4
)

score_snow = koof_t_snow * koof_w_snow * koof_o_snow / 1.617
snow_active = (t >= -15) & (t <= 2) & (snow > 0)
score_snow = np.where(snow_active, score_snow, 0.0)


#гололёд
koof_t_ice = np.select(
    [(t <= 0) & (t > -5), (t <= -5) & (t >= -10)],
    [2, 1], 
    default=0
)

koof_h_ice = np.select(
    [h < 50, h < 60, h < 70, h < 80, h < 90],
    [0, 1, 2, 3, 4],
    default=5
)

koof_h_eff = np.where(
    (koof_t_ice == 1) & (koof_h_ice == 1), 0,
    np.where(koof_t_ice == 2, koof_h_ice, np.clip(koof_h_ice - 1, 0, None))
)

score_ice = np.where(
    koof_t_ice == 0,
    1.0,
    koof_h_eff * koof_t_ice
) / 2.0


#изморозь

winter_eval['temp_prev'] = winter_eval.groupby('res_id')['temp_day'].shift(1).fillna(winter_eval['temp_day'])

t2 = winter_eval['temp_prev'].values
dt_abs = np.abs(dt)

koof_t_rime = np.select(
    [dt_abs < 7, dt_abs < 8, dt_abs <= 11],
    [1, 2, 3],
    default=4
)

score_rime = np.where(
    t2 <= -5,
    (koof_t_rime * winter_eval['izmoroz_koef']) / 3.0,
    0.0
)

winter_eval['baseline_pred'] = ((score_snow >= 2.0) | (score_ice >= 2.0) | (score_rime >= 2.0)).astype(int)

#ветер + осадки

deg = summer_eval['wind_deg'].values
conditions = [
    (deg >= 33.75) & (deg < 56.25),   # NE (С-В)
    (deg >= 56.25) & (deg < 123.75),  # E (В)
    (deg >= 123.75) & (deg < 168.75), # SE (Ю-В)
    (deg >= 168.75) & (deg < 213.75), # S (Ю)
    (deg >= 213.75) & (deg < 258.75), # SW (Ю-З)
    (deg >= 258.75) & (deg < 303.75), # W (З)
    (deg >= 303.75) & (deg < 326.25), # NW (С-З)
]
choices = ['Северо-восток', 'Восток', 'Юго-восток', 'Юг', 'Юго-запад', 'Запад', 'Северо-запад']
summer_eval['wind_dir_name'] = np.select(conditions, choices, default='Север')

summer_eval = summer_eval.merge(
    coef_one[['filial', 'res', 'wind_direction', 'koof_wind']],
    left_on=['filial', 'res', 'wind_dir_name'],
    right_on=['filial', 'res', 'wind_direction'],
    how='left'
).fillna({'koof_wind': 1.0})

summer_eval = summer_eval.merge(
    coef_four[['filial', 'res', 'min_wind', 'max_wind', 'koof_wind2']],
    on=['filial', 'res'],
    how='left'
)

summer_eval = summer_eval[
    ((summer_eval['wind_speed'] >= summer_eval['min_wind']) & (summer_eval['wind_speed'] <= summer_eval['max_wind'])) | 
    (summer_eval['min_wind'].isnull())
]

summer_eval['koof_wind2'] = summer_eval['koof_wind2'].fillna(1.0)
summer_eval['precip'] = summer_eval['rain'].fillna(0) + summer_eval['snow'].fillna(0)

summer_eval = summer_eval.merge(
    coef_vol[['filial', 'res', 'min_value', 'max_value', 'koof_vol']],
    on=['filial', 'res'],
    how='left'
)


summer_eval = summer_eval[
    ((summer_eval['precip'] >= summer_eval['min_value']) & (summer_eval['precip'] <= summer_eval['max_value'])) | 
    (summer_eval['min_value'].isnull())
]
summer_eval['koof_vol'] = summer_eval['koof_vol'].fillna(1.0)


summer_eval['summer_score'] = summer_eval['koof_wind'] * summer_eval['koof_wind2'] * summer_eval['koof_vol']
summer_eval['baseline_pred'] = (summer_eval['summer_score'] >= 3.3).astype(int)


eval_cols = ['res_id', 'forecast_date', 'target', 'baseline_pred']
baseline_results = pd.concat([
    winter_eval[eval_cols],
    summer_eval[eval_cols]
]).sort_values(['res_id', 'forecast_date']).reset_index(drop=True)


from sklearn.metrics import classification_report, roc_auc_score, fbeta_score

y_true_baseline = baseline_results['target'].values
y_pred_baseline = baseline_results['baseline_pred'].values

print("\nМетрики старой системы:\n")
print(classification_report(y_true_baseline, y_pred_baseline, target_names=["Нет аварии", "Авария"]))
print(f"ROC-AUC бейзлайна: {roc_auc_score(y_true_baseline, y_pred_baseline):.4f}")
print(f"F2-score бейзлайна: {fbeta_score(y_true_baseline, y_pred_baseline, beta=2):.4f}")

#Обучение ML модели

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

cat_cols = ['city_name', 'weather_main']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

param_grid = {
    'learning_rate' : [0.03, 0.05, 0.1],
    'max_depth' : [3, 5, 8],
    'max_iter' : [100, 150],
    'l2_regularization': [0.1, 1.0, 10.0]
}

model = HistGradientBoostingClassifier(
    class_weight='balanced',
    categorical_features=cat_cols,
    random_state=42
)


# Отфильтруем признаки для обучения (без даты и утечки outage_count)
features = [col for col in X_train.columns if col not in ['forecast_date', 'outage_count']]

print('Запуск gridsearch')
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train[features], y_train)

print(f'Лучшие параметры: {grid.best_params_}')
best_model = grid.best_estimator_

preds = best_model.predict_proba(X_train[features])[:,1]
best_thresh = 0.5
best_f2 = 0

for thresh in np.linspace(0.01, 0.99, 50):
    f2 = fbeta_score(y_train, (preds>=thresh).astype(int), beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_thresh = thresh

test_preds = best_model.predict_proba(X_test[features])[:,1]
y_pred_ml = (test_preds >= best_thresh).astype(int)

print("\nМетрики ML-модели:\n")
print(classification_report(y_test, y_pred_ml, target_names=["Нет аварии", "Авария"]))
print(f"ROC-AUC ML-модели: {roc_auc_score(y_test, test_preds):.4f}")
print(f"F2-score ML-модели: {fbeta_score(y_test, y_pred_ml, beta=2):.4f}")