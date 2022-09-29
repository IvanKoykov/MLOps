# MLOps
Проект по MLOps.
Что сделано:
1)Загрузка исходного датасета на clearml.
2) Скачивание и Валидация данных в исходном датасете (people)
3) Подготовка данных и преобразование данных и загрузка нового датасета на clearml. Новый датасет разделен на mask_people_2, каждый из которых данные разделены на train/val/test и могут быть любого размера.
4) Обучение модели.С использованием модели из PyTorch с логгированием через tensorboard. После обучения модель выгружется в .onnx формате как артефакт.
5)При выполнении push в github происходит проверка кода через flake8.

Что планируется сделать:
1) Инференс и мониторинг модели с использованием clearml-serving.
2) Сохранение лучшей метрики (коэффициент Дайса или IOU) для будующей загрузки из clearml и сравнения с новыми моделями. 
3)Выбор более эффективной модели.
