from FortuneTeller import *

# Загрузка данных
sales = pd.read_csv('data/shop_sales.csv')
calendar = pd.read_csv('data/shop_sales_dates.csv')
prices = pd.read_csv('data/shop_sales_prices.csv')

# Преобразование полей в удобные типы
calendar['date'] = pd.to_datetime(calendar['date'])  # Преобразование даты в объект datetime
sales['date_id'] = sales['date_id'].astype(int)  # Преобразование идентификатора даты в int

# Шаг 1: Объединяем sales с calendar по полю `date_id`
sales_calendar_merged = pd.merge(sales, calendar, how='left', on='date_id')

# Шаг 2: Объединяем с prices по полям `item_id`, `store_id` и `wm_yr_wk`
df = pd.merge(
    sales_calendar_merged, 
    prices, 
    how='left', 
    on=['item_id', 'store_id', 'wm_yr_wk']
)

# Assuming your dataset is loaded into a DataFrame called "df"
# Parse the 'date' column to ensure it's recognized as a datetime type
df['date'] = pd.to_datetime(df['date'])

f = FortuneTeller(df)
f.test_train()
f.fit()
f.predict()
f.save()
f.evaluate()
f.show()

