from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, log_loss, mean_squared_error, r2_score
)


# 전체 샘플 중 맞춘 비율 - Accuracy(TP + TN) / (TP + FP + FN + TN)
# 모델이 긍정(Positive)이라고 예측한 것 중 진짜 Positive - Precision (TP) / (TP + FP)
# 실제 Positive 중 모델이 올바르게 맞춘 비율 - Recall (TP) / (TP + FN)
# Precision과 Recall의 조화평균, 불균형 데이터셋에서 중요 - F1 = 2*(Precision * Recall) / (Precision + Recall)
# Confusion Matrix -> TP, TN, FP, FN
# Classification Report -> Precision, Recall, F1, Support를 한 번에 보여줌 + 다중 클래스 분류에서 많이 사용
# ROC, AUC
# Log Loss -> Cross Entropy Loss -> -y_i*log(p_i) => 실제 레이블과 다른 값을 예측할 수록 로스가 커짐을 확인할 수 있음
# Mean Squared Error(MSE) -> 회귀 문제에서 가장 많이 사용
# R^2 Score 결정계수 -> 회귀에서 사용하며 1.0 ~ -1.0 (양수에 가까울 수록 예측 성공률 증가)


y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
y_prob = [0.2, 0.8, 0.4, 0.3, 0.9]


'''
image, label = image.to(device), label.to(device)
pred = model(image)
'''
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_prob))
print("Log Loss:", log_loss(y_true, y_prob))

# Regression example
y_true_reg = [3.0, 2.5, 4.0]
y_pred_reg = [2.8, 2.7, 3.9]

print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
print("R2 Score:", r2_score(y_true_reg, y_pred_reg))
