import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Data preparation
# =========================

df = pd.read_csv('data.csv')

# Нормализуем названия колонок
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Находим строковые колонки
strings = list(df.dtypes[df.dtypes == 'object'].index)

# Нормализуем строковые значения
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# =========================
# EDA
# =========================

# Смотрим значения в колонках
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()


# Смотрим распределение цены
sns.histplot(df.msrp, bins=50)
plt.show()

# Смотрим основную часть распределения
sns.histplot(df.msrp[df.msrp < 100000], bins=50)
plt.show()

# Логарифмируем цену
price_logs = np.log1p(df.msrp)

sns.histplot(price_logs, bins=50)
plt.show()

# Проверяем пропуски
print(df.isnull().sum())


# =========================
# Validation framework
# =========================

# Размер всего датасета
n = len(df)

# 20% validation
n_val = int(n * 0.2)

# 20% test
n_test = int(n * 0.2)

# Остальное — train (~60%)
n_train = n - n_val - n_test

print(n, n_train, n_val, n_test)


# =========================
# Shuffle
# =========================

# Фиксируем случайность для воспроизводимости
np.random.seed(2)

# Создаём индексы строк
idx = np.arange(n)

# Перемешиваем индексы
np.random.shuffle(idx)


# =========================
# Train / Validation / Test
# =========================

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]


# Сбрасываем старые индексы
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# Проверяем размеры
print(
    len(df_train),
    len(df_val),
    len(df_test)
)


# =========================
# Target y
# =========================

# Берём MSRP и применяем log1p
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)


# =========================
# Удаляем target из features
# =========================

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']