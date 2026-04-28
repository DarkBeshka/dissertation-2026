# Установка пакетов (если не установлены)
install.packages(c("tidyverse", "corrplot"))

# Подключение библиотек
library(tidyverse)
library(corrplot)

# Загрузка данных
data <- read.csv("master_table_russia_2001_01_long_sample.csv")

# Быстрый просмотр структуры
str(data)

# Оставляем только числовые признаки
numeric_data <- data %>%
  select(where(is.numeric))

# Проверка на пропуски
colSums(is.na(numeric_data))

# При необходимости можно удалить строки с NA
numeric_data_clean <- na.omit(numeric_data)

# Расчёт корреляционной матрицы (Пирсон)
cor_matrix <- cor(numeric_data_clean, method = "pearson")

# Вывод матрицы
print(cor_matrix)

# Визуализация
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.7,
         number.cex = 0.6,
         addCoef.col = "black")