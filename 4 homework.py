
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Вероятность класса 1: пользователь купит курс
y_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)
print('Q2:', round(auc, 3))


# Q3 и Q4. Проверяем разные пороги от 0.00 до 0.99
thresholds = np.arange(0.0, 1.0, 0.01)
precisions = []
recalls = []
f1_scores = []

for t in thresholds:
    predicted = (y_pred >= t).astype(int)

    p = precision_score(y_val, predicted, zero_division=0)
    r = recall_score(y_val, predicted, zero_division=0)
    f1 = f1_score(y_val, predicted, zero_division=0)

    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f1)

# Q3: порог, где precision и recall ближе всего друг к другу
best_threshold = 0
smallest_diff = 10

for i in range(len(thresholds)):
    p = precisions[i]
    r = recalls[i]

    if p == 0 and r == 0:
        continue

    diff = abs(p - r)

    if diff < smallest_diff:
        smallest_diff = diff
        best_threshold = thresholds[i]

print('Q3:', round(best_threshold, 2))

# Строим график precision и recall
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.savefig('precision_recall.png')
plt.close()

# Q4: ищем порог с максимальным F1
best_f1 = max(f1_scores)
best_f1_index = f1_scores.index(best_f1)
print('Q4:', round(thresholds[best_f1_index], 2))


# Q5. Проверка модели на пяти разных частях данных (5-Fold CV)
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
cv_scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    fold_train = df_full_train.iloc[train_idx]
    fold_val = df_full_train.iloc[val_idx]

    y_fold_train = fold_train['converted'].values
    y_fold_val = fold_val['converted'].values

    X_fold_train_df = fold_train.drop(columns=['converted'])
    X_fold_val_df = fold_val.drop(columns=['converted'])

    dv = DictVectorizer(sparse=False)
    X_fold_train = dv.fit_transform(X_fold_train_df.to_dict(orient='records'))
    X_fold_val = dv.transform(X_fold_val_df.to_dict(orient='records'))

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_fold_train, y_fold_train)

    predictions = model.predict_proba(X_fold_val)[:, 1]
    score = roc_auc_score(y_fold_val, predictions)
    cv_scores.append(score)

print('Q5:', round(np.std(cv_scores), 3))


# Q6. Выбираем лучший параметр C с помощью 5-Fold CV
C_values = [0.000001, 0.001, 1]
best_C = None
best_mean = -1
best_std = 10

for C in C_values:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        fold_train = df_full_train.iloc[train_idx]
        fold_val = df_full_train.iloc[val_idx]

        y_fold_train = fold_train['converted'].values
        y_fold_val = fold_val['converted'].values

        X_fold_train_df = fold_train.drop(columns=['converted'])
        X_fold_val_df = fold_val.drop(columns=['converted'])

        dv = DictVectorizer(sparse=False)
        X_fold_train = dv.fit_transform(X_fold_train_df.to_dict(orient='records'))
        X_fold_val = dv.transform(X_fold_val_df.to_dict(orient='records'))

        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_fold_train, y_fold_train)

        predictions = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, predictions)
        scores.append(score)

    mean_auc = round(np.mean(scores), 3)
    std_auc = round(np.std(scores), 3)
    print('C =', C, 'mean =', mean_auc, 'std =', std_auc)

    if mean_auc > best_mean or (mean_auc == best_mean and std_auc < best_std):
        best_mean = mean_auc
        best_std = std_auc