from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import time

try:
    # Đọc dữ liệu từ file CSV với mã hóa 'utf-8'
    data = pd.read_csv('D:\\fake news\\fake-news\\data4.csv', encoding='utf-8')
except UnicodeDecodeError:
    # Nếu gặp lỗi mã hóa, thử với mã hóa 'latin1'
    data = pd.read_csv('D:\\fake news\\fake-news\\data4.csv', encoding='latin1')

# Chia dữ liệu thành features (đặc trưng) và labels (nhãn)
X = data.drop('Label', axis=1)  # Bỏ cột nhãn khỏi features
y = data['Label']

# Khởi tạo mô hình cây quyết định
decision_tree = DecisionTreeClassifier()

# Tính độ chính xác trung bình của cây quyết định bằng cross-validation
accuracy_decision_tree = cross_val_score(decision_tree, X, y, cv=5).mean()

# Khởi tạo mô hình Random Forest
random_forest = RandomForestClassifier(n_estimators=100)

# Đo thời gian thực thi và tính độ chính xác trung bình của Random Forest bằng cross-validation
start_time_rf = time.time()
accuracy_random_forest = cross_val_score(random_forest, X, y, cv=5).mean()
end_time_rf = time.time()
execution_time_rf = end_time_rf - start_time_rf

# Khởi tạo mô hình Bagging
# bagging = BaggingClassifier(base_estimator=decision_tree, n_estimators=100)

# Đo thời gian thực thi và tính độ chính xác trung bình của Bagging bằng cross-validation
# start_time_bagging = time.time()
# accuracy_bagging = cross_val_score(bagging, X, y, cv=5).mean()
# end_time_bagging = time.time()
# execution_time_bagging = end_time_bagging - start_time_bagging

# In kết quả
print("Độ chính xác trung bình của cây quyết định:", accuracy_decision_tree)
print("Độ chính xác trung bình của Random Forest:", accuracy_random_forest)
# print("Độ chính xác trung bình của Bagging:", accuracy_bagging)
print("Thời gian thực thi của Random Forest:", execution_time_rf)
# print("Thời gian thực thi của Bagging:", execution_time_bagging)
