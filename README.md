# Whale Classification

## Overview

Задача представляет собой [kaggle-соревнование](https://www.kaggle.com/competitions/humpback-whale-identification/overview) по классификации китов на изображениях.

**Цель работы:** решить данную задачу, используя подход `mertic learning`, а также разработать демо-сервис для инференса изображений.

## Dataset

Датасет состоит из двух папок с изображениями китов:
* train - 25361 изображений
* test - 7960 изображений

Также приложен файл `train.csv`, который содержит информацию о классе кита на каждом изображении. Всего в наборе данных представлено **5005 уникальных видов китов**, из которых обнаружено:
* чаще всего встречается класс `new_whale`- 9664 из 25361 изображений
* 2073 класса содержат только одно изображение

Для обучения и валидации пайплайна набор данных был разделен на две части, причем в обучающую выборку вошли:
* все классы (2073 класса), содержащие только одно изображение на класс;
* 7731 изображений из класса new_whale, остальные 1933 - в валидационную выборку;
* все остальные классы были разделены с помощью метода StratifiedKFold (тут понял, что ошибся, так как изображения разделились поровну между train и val выборками - 2931 уникальных классов по 6812 изображений на выборку).

## Models

К настоящему времени в качестве энкодеров для получения эмбеддингов были опробованы следующие модели:
* непредобученный `resnet18` из torchvision;
* предобученный `ViT` (Visual Transformer) из hugging-face (веса - `google/vit-base-patch16-224`)




path = https://drive.google.com/file/d/1NBph6h8somnUr5nZk8jwLYotbYryhE2v/view?usp=share_link

## Metrics
