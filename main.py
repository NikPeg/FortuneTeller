import pandas as pd
import os
from FortuneTeller import FortuneTeller

def main():
    print("Добро пожаловать в FortuneTeller CLI!")
    print("Вы хотите использовать готовую модель или обучить новую?")
    print("1: Использовать готовую модель")
    print("2: Обучить новую модель")
    
    while True:
        choice = input("Введите 1 или 2: ").strip()
        if choice in ['1', '2']:
            break
        else:
            print("Неверный ввод. Пожалуйста, введите 1 или 2.")
    
    if choice == '1':
        # Использование готовой модели
        f = FortuneTeller(None)  # Пустой датасет, так как будем загружать готовую модель

        # Спрашиваем у пользователя путь к тестовому датасету или оставляем стандартный
        test_dataset_path = input("Введите путь к тестовому датасету (оставьте пустым для использования 'data/test.csv'): ").strip()
        if not test_dataset_path:
            test_dataset_path = 'data/test.csv'

        try:
            f.load()  # Загрузка готовой модели
            print("Готовая модель успешно загружена!")

            # Проверим наличие указанного пользователем файла
            import os
            if os.path.exists(test_dataset_path):
                print(f"Будет использоваться тестовый файл: {test_dataset_path}")
            else:
                print(f"Ошибка: файл {test_dataset_path} не найден.")
                return
            # Предсказываем результаты по обученной модели
            test = pd.read_csv(test_dataset_path)
            f.predict(test)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return
    
    elif choice == '2':
        # Обучение новой модели
        print("Введите названия файлов с данными (нажмите Enter для использования файлов по умолчанию):")
        sales_file = input("Файл с продажами (по умолчанию 'data/shop_sales.csv'): ").strip() or 'data/shop_sales.csv'
        calendar_file = input("Файл с календарем (по умолчанию 'data/shop_sales_dates.csv'): ").strip() or 'data/shop_sales_dates.csv'
        prices_file = input("Файл с ценами (по умолчанию 'data/shop_sales_prices.csv'): ").strip() or 'data/shop_sales_prices.csv'
        
        try:
            # Проверяем существование файлов
            if not os.path.exists(sales_file):
                raise FileNotFoundError(f"Файл '{sales_file}' не найден.")
            if not os.path.exists(calendar_file):
                raise FileNotFoundError(f"Файл '{calendar_file}' не найден.")
            if not os.path.exists(prices_file):
                raise FileNotFoundError(f"Файл '{prices_file}' не найден.")
            
            # Загружаем данные
            sales = pd.read_csv(sales_file)
            calendar = pd.read_csv(calendar_file)
            prices = pd.read_csv(prices_file)

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

            # Преобразование даты в объект datetime
            df['date'] = pd.to_datetime(df['date'])

            # Создаем объект FortuneTeller
            f = FortuneTeller(df)
            print("Данные успешно загружены и обработаны!")

            # Выполняем обучение и прогнозирование
            f.test_train()
            f.fit()
            f.predict()
            f.save()
            print("Модель успешно обучена и сохранена!")
        
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return

    # Оценка и визуализация модели
    try:
        f.evaluate()
        f.show()
    except Exception as e:
        print(f"Произошла ошибка при оценке или визуализации: {e}")
    
    print("Работа завершена!")

if __name__ == "__main__":
    main()

