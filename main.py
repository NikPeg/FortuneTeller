import pandas as pd
from FortuneTeller import FortuneTeller

def main():
    print("Добро пожаловать в программу предсказания продаж с использованием FortuneTeller!")
    
    # Шаг 1: Загрузка данных
    print("\nДля начала загрузим необходимые данные. Введите названия файлов (или оставьте пустым для выхода из программы).")
    
    sales_file = input("Введите путь к файлу с данными о продажах (например, 'data/shop_sales.csv'): ")
    if not sales_file:
        print("Выход из программы.")
        return
    
    calendar_file = input("Введите путь к файлу с календарными данными (например, 'data/shop_sales_dates.csv'): ")
    if not calendar_file:
        print("Выход из программы.")
        return
    
    prices_file = input("Введите путь к файлу с данными о ценах (например, 'data/shop_sales_prices.csv'): ")
    if not prices_file:
        print("Выход из программы.")
        return

    try:
        # Загрузка данных
        sales = pd.read_csv(sales_file)
        calendar = pd.read_csv(calendar_file)
        prices = pd.read_csv(prices_file)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return

    # Шаг 2: Предобработка данных
    print("\nПреобразуем данные...")
    try:
        calendar['date'] = pd.to_datetime(calendar['date'])  # Преобразование дат
        sales['date_id'] = sales['date_id'].astype(int)  # Преобразование идентификатора даты в int
        
        # Объединение данных
        sales_calendar_merged = pd.merge(sales, calendar, how='left', on='date_id')
        df = pd.merge(
            sales_calendar_merged, 
            prices, 
            how='left', 
            on=['item_id', 'store_id', 'wm_yr_wk']
        )
        
        df['date'] = pd.to_datetime(df['date'])  # Преобразование к типу datetime
    except Exception as e:
        print(f"Ошибка при предобработке данных: {e}")
        return

    print("Данные подготовлены!")

    # Шаг 3: Создание объекта FortuneTeller
    print("\nСоздаем объект FortuneTeller...")
    try:
        f = FortuneTeller(df)
    except Exception as e:
        print(f"Ошибка при создании FortuneTeller: {e}")
        return

    # Шаг 4: Тренировка и прогнозирование
    while True:
        print("\nЧто вы хотите сделать?")
        print("1. Разделить данные на тестовую и обучающую выборки")
        print("2. Обучить модель")
        print("3. Построить прогнозы")
        print("4. Оценить качество модели")
        print("5. Показать результаты")
        print("6. Выйти")

        choice = input("Введите номер действия: ")

        if choice == "1":
            try:
                f.test_train()
                print("Данные успешно разделены на тестовую и обучающую выборки!")
            except Exception as e:
                print(f"Ошибка при разделении данных: {e}")
        elif choice == "2":
            try:
                f.fit()
                print("Модель успешно обучена!")
            except Exception as e:
                print(f"Ошибка при обучении модели: {e}")
        elif choice == "3":
            try:
                f.predict()
                print("Прогнозы успешно построены!")
            except Exception as e:
                print(f"Ошибка при построении прогнозов: {e}")
        elif choice == "4":
            try:
                f.evaluate()
                print("Качество модели успешно оценено!")
            except Exception as e:
                print(f"Ошибка при оценке модели: {e}")
        elif choice == "5":
            try:
                f.show()
                print("Результаты успешно показаны!")
            except Exception as e:
                print(f"Ошибка при отображении результатов: {e}")
        elif choice == "6":
            print("Выходим из программы. До свидания!")
            break
        else:
            print("Неверный выбор. Пожалуйста, введите номер действия от 1 до 6.")

if __name__ == "__main__":
    main()

