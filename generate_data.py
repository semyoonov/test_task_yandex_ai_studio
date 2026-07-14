import pandas as pd

weather = pd.read_csv('dataset/weather/daily_weather.csv')
res_reference = pd.read_csv('dataset/assets/res_reference.csv')
events = pd.read_csv('dataset/target/outage_events.csv')

events['date'] = pd.to_datetime(events['dt_start']).dt.date
weather = weather.drop_duplicates(subset=['city_id', 'forecast_date'])
weather['forecast_date'] = pd.to_datetime(weather['forecast_date'])

all_dates = weather['forecast_date'].unique()
all_res = res_reference['res_id'].unique()

grid = pd.MultiIndex.from_product(
    [all_res, all_dates],
    names=['res_id', 'forecast_date']
).to_frame().reset_index(drop=True)

events_daily = events.groupby(['res_id', 'date']).size().reset_index(name='outage_count')
grid['date_only'] = grid['forecast_date'].dt.date


dataset = grid.merge(
    events_daily,
    left_on=['res_id', 'date_only'],
    right_on=['res_id', 'date'],
    how='left'
)

dataset['target'] = (dataset['outage_count'].fillna(0) > 0).astype(int)
dataset = dataset.drop(columns=['date_only', 'date'])


dataset = dataset.merge(
    res_reference[['res_id', 'city_id', 'customers_count', 'forest_share', 'sip_share', 'mean_age']], 
    on='res_id', 
    how='left'
)

dataset = dataset.merge(
    weather,
    on=['city_id', 'forecast_date'],
    how='left'
)

telemetry = pd.read_csv('dataset/telemetry/feeder_load_daily.csv')
telemetry['forecast_date'] = pd.to_datetime(telemetry['forecast_date'])

telemetry = telemetry.sort_values(['res_id', 'forecast_date']).reset_index(drop=True)

telemetry['load_mw'] = telemetry.groupby('res_id')['load_mw'].shift(1)
telemetry['voltage_dips'] = telemetry.groupby('res_id')['voltage_dips'].shift(1)
telemetry['mean_feeder_temp_c'] = telemetry.groupby('res_id')['mean_feeder_temp_c'].shift(1)

for col in ['load_mw', 'voltage_dips', 'mean_feeder_temp_c']:
    telemetry[col] = telemetry.groupby('res_id')[col].transform(lambda x: x.fillna(x.median()))
    telemetry[col] = telemetry[col].fillna(telemetry[col].median())

dataset = dataset.merge(
    telemetry,
    on=['res_id', 'forecast_date'],
    how='left'
)

dataset.to_csv('dataset.csv', index=False)
print(f'Датасет создан!\nФайл: dataset.csv\nРазмер: {len(dataset)} строк')