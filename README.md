# FortuneTeller
FortuneTeller CLI — это интерфейс командной строки для работы с прогнозами на основе данных продаж. С помощью этого скрипта вы можете либо загрузить готовую модель, либо обучить новую на предоставленных данных.
Пример использования полученного класса приведен в файле example.py и example.ipynb.
    
## Требования  
- Python версии 3.7 и выше  
- Виртуальная среда (рекомендуется)  
- Модуль FortuneTeller (должен находиться в той же директории, что и скрипт)  
## Установка и запуск
Следуйте этим шагам, чтобы установить и запустить скрипт:  
  
Клонирование репозитория или копирование файлов:
  
Скопируйте скрипт main.py, модуль FortuneTeller и все необходимые данные (shop_sales.csv, shop_sales_dates.csv, shop_sales_prices.csv) в одну директорию.
  
Создание и активация виртуальной среды:
  
На вашем терминале выполните следующие команды:
  
```# Создать виртуальную среду
python -m venv venv
  
# Активировать виртуальную среду
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate```
Установка библиотек:

Убедитесь, что файл requirements.txt находится в той же директории, и выполните команду:

```pip install -r requirements.txt```
Запуск скрипта:
  
Выполните скрипт main.py следующим образом:
  
python main.py
## Как пользоваться
После запуска скрипта вы увидите меню, чтобы выбрать действие:
  
Использовать готовую модель:
  
Выберите 1 и нажмите Enter.  
Программа загрузит обученную модель (если она существует).  
Она оценит модель и покажет результаты прогноза.  
Обучить новую модель:  
  
Выберите 2 и нажмите Enter.  
Затем введите пути к файлам данных (или оставьте поля пустыми, чтобы использовать значения по умолчанию):  
Файл с продажами (shop_sales.csv)  
Файл с календарем (shop_sales_dates.csv)  
Файл с ценами (shop_sales_prices.csv)  
Программа загрузит данные, обработает их, создаст новую модель, обучит её, сохранит и предоставит прогнозы.  
После выполнения действия (обучения или работы с готовой моделью) программа также оценит модель, покажет результаты прогноза и визуализации.  

## Файлы по умолчанию
Файл с продажами: data/shop_sales.csv  
Файл с календарем: data/shop_sales_dates.csv  
Файл с ценами: data/shop_sales_prices.csv  
Если файлов нет, скрипт выдаст ошибку. Вы можете указать пути к другим файлам вручную.  
  
## Примечания
Все результаты обучения автоматически сохраняются в методе f.save(). Это означает, что вы сможете использовать эту модель в будущем с помощью выбора "Использовать готовую модель".
  
Обработка данных выполняется перед обучением, так что ваш набор данных должен быть корректно подготовлен.
## Примеры использования
Пример 1: Использование готовой модели  
Добро пожаловать в FortuneTeller CLI!  
Вы хотите использовать готовую модель или обучить новую?  
1: Использовать готовую модель  
2: Обучить новую модель  
Введите 1 или 2: 1  
Готовая модель успешно загружена!  
Оценка модели...  
Прогнозирование выполнено!  
Пример 2: Обучение новой модели  
Добро пожаловать в FortuneTeller CLI!  
Вы хотите использовать готовую модель или обучить новую?  
1: Использовать готовую модель  
2: Обучить новую модель  
Введите 1 или 2: 2  
Введите названия файлов с данными (нажмите Enter для использования файлов по умолчанию):  
Файл с продажами (по умолчанию 'data/shop_sales.csv'):  
Файл с календарем (по умолчанию 'data/shop_sales_dates.csv'):  
Файл с ценами (по умолчанию 'data/shop_sales_prices.csv'):  
Данные успешно загружены и обработаны!  
Модель обучается...  
Прогнозирование выполнено и модель сохранена!  
Ошибки и их решения  
Файл не найден: Убедитесь, что указали корректные пути к файлам, или поместите файлы в папку по умолчанию (data/).  
Библиотека не установлена: Установите зависимости через pip install -r requirements.txt.  
Готовая модель не найдена: Если вы еще не обучали модель, выберите второй вариант — обучение новой модели.  
## Лицензия
Данный проект предоставляется "как есть". Используйте на свой страх и риск.
