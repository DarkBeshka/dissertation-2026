install.packages(c("tidyverse", "corrplot"))

library(tidyverse)
library(corrplot)

data <- read.csv("master_table_russia_2001_01_long_sample.csv")

str(data)

# Оставляем только числовые признаки
numeric_data <- data %>%
  select(where(is.numeric))

colSums(is.na(numeric_data))
numeric_data_clean <- na.omit(numeric_data)

# Расчёт корреляционной матрицы (Пирсон)
cor_matrix <- cor(numeric_data_clean, method = "pearson")

print(cor_matrix)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.7,
         number.cex = 0.6,
         addCoef.col = "black")
