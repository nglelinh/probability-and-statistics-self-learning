---
layout: post
title: 01-10-00 Quy Trình Thực Hành Machine Learning
chapter: "01"
order: 10
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: required
---

Trong thực tế công nghiệp, phần lớn các dự án học máy thất bại **không phải vì chọn nhầm thuật toán** — mà vì một lỗi thầm lặng chết người trong quy trình dữ liệu: **rò rỉ dữ liệu (data leakage)**, chuẩn hóa trước khi chia tập kiểm tra, hay quên mất một mô hình cơ sở đơn giản để đối chứng. Một thuật toán bình thường chạy trên một quy trình sạch sẽ luôn đánh bại một mô hình tối tân chạy trên một quy trình cẩu thả.

Bài học này ghép nối mọi kỹ thuật rời rạc thành một **quy trình thực hành Machine Learning chuẩn mực từ đầu đến cuối**, giúp bạn đi từ bảng dữ liệu thô đầy lỗ hổng đến một mô hình đáng tin cậy, giải thích được và sẵn sàng kiểm chứng ngoài đời thực.

---

## Bản đồ kiến thức nền: Quy trình Machine Learning chuẩn khoa học gồm những trụ cột nào?

Một quy trình ML nghiêm túc thực chất là một **thí nghiệm thống kê có kiểm soát**, đứng trên **bốn trụ cột kỷ luật**:

1. **Khám phá & Làm sạch (EDA & Preprocessing):** Hiểu phân phối, phát hiện giá trị thiếu, ngoại lai và thiên lệch dữ liệu trước khi đụng đến bất kỳ mô hình nào.
2. **Cách ly tuyệt đối tập kiểm tra (Leakage-Free Split & Pipeline):** Mọi phép biến đổi học từ dữ liệu (điền trung vị, chuẩn hóa trung bình/độ lệch chuẩn, mã hóa nhãn, chọn đặc trưng) chỉ được phép *khớp* (`fit`) trên tập huấn luyện rồi *áp dụng* (`transform`) sang tập kiểm định/kiểm tra.
3. **Đi từ Baseline đến Tinh chỉnh:** Luôn bắt đầu bằng mô hình đơn giản nhất (hồi quy tuyến tính, logistic, cây nông) để đặt chuẩn so sánh, rồi mới tăng độ phức tạp và tinh chỉnh siêu tham số trên tập Validation.
4. **Đánh giá cuối cùng & Diễn giải:** Tập Test chỉ được mở khóa **đúng một lần duy nhất** ở bước cuối cùng; sau đó kiểm tra mức độ quan trọng đặc trưng và lỗi trên từng nhóm con để chắc chắn mô hình học đúng quy luật chứ không học mẹo.

---

## Quy trình Machine Learning từ đầu đến cuối

Một dự án học máy thực tế thường gồm 8 bước chính:

1. **Hiểu bài toán và khám phá dữ liệu** (Problem Understanding & EDA)
2. **Tiền xử lý dữ liệu** (Data Preprocessing)
3. **Kỹ thuật đặc trưng** (Feature Engineering)
4. **Chia tập dữ liệu** (Train/Validation/Test Split)
5. **Xây dựng mô hình cơ sở** (Baseline Model)
6. **Lựa chọn mô hình và tinh chỉnh siêu tham số** (Model Selection & Hyperparameter Tuning)
7. **Đánh giá cuối cùng** (Final Evaluation)
8. **Diễn giải mô hình** (Model Interpretation)

Dưới đây là phần cài đặt minh họa từng bước trên bộ dữ liệu California Housing.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

## Bước 1: Hiểu bài toán và khám phá dữ liệu (EDA)

```python
from sklearn.datasets import fetch_california_housing

def load_and_explore_data():
    """
    Load dữ liệu và thực hiện EDA cơ bản
    """
    # Load California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    print("=" * 60)
    print("THÔNG TIN DỮ LIỆU")
    print("=" * 60)
    print(f"Số samples: {len(df)}")
    print(f"Số features: {len(df.columns) - 1}")
    print(f"\nTarget variable: {housing.target_names[0]}")
    print(f"\nFeatures:")
    for i, name in enumerate(housing.feature_names):
        print(f"  {i+1}. {name}")
    
    print("\n" + "=" * 60)
    print("THỐNG KÊ MÔ TẢ")
    print("=" * 60)
    print(df.describe())
    
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution of target
    axes[0, 0].hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Median House Value (100k$)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Target Variable')
    
    # Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1],
               cbar_kws={'label': 'Correlation'})
    axes[0, 1].set_title('Correlation Matrix')
    
    # Scatter: MedInc vs Target
    axes[1, 0].scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Median Income')
    axes[1, 0].set_ylabel('Median House Value')
    axes[1, 0].set_title('Income vs House Value')
    
    # Boxplot: AveRooms
    axes[1, 1].boxplot(df['AveRooms'])
    axes[1, 1].set_ylabel('Average Rooms')
    axes[1, 1].set_title('Distribution of Average Rooms')
    
    plt.tight_layout()
    # plt.show()
    
    return df

# df = load_and_explore_data()
```

## Bước 2: Tiền xử lý dữ liệu (Data Preprocessing)

```python
def preprocess_data(df):
    """
    Chuẩn bị dữ liệu: xử lý outliers, scaling
    """
    # Tách features và target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Xử lý outliers (IQR method)
    def remove_outliers(data, column, threshold=3):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return data[(data[column] >= lower) & (data[column] <= upper)]
    
    # Loại bỏ outliers cho một số features
    df_clean = df.copy()
    for col in ['AveRooms', 'AveBedrms']:
        df_clean = remove_outliers(df_clean, col)
    
    print(f"Số samples sau khi loại outliers: {len(df_clean)} (giảm {len(df) - len(df_clean)})")
    
    X_clean = df_clean.drop('MedHouseVal', axis=1)
    y_clean = df_clean['MedHouseVal']
    
    return X_clean, y_clean

# X, y = preprocess_data(df)
```

## Bước 3: Kỹ thuật đặc trưng (Feature Engineering)

```python
def create_features(X):
    """
    Tạo features mới từ features hiện có
    """
    X_new = X.copy()
    
    # Tạo features mới
    X_new['RoomsPerHousehold'] = X['AveRooms'] / X['AveOccup']
    X_new['BedroomsPerRoom'] = X['AveBedrms'] / X['AveRooms']
    X_new['PopulationPerHousehold'] = X['Population'] / X['AveOccup']
    
    # Log transform cho features skewed
    X_new['LogPopulation'] = np.log1p(X['Population'])
    X_new['LogMedInc'] = np.log1p(X['MedInc'])
    
    print("Features mới đã tạo:")
    print("  - RoomsPerHousehold")
    print("  - BedroomsPerRoom")
    print("  - PopulationPerHousehold")
    print("  - LogPopulation")
    print("  - LogMedInc")
    
    return X_new

# X_engineered = create_features(X)
```

## Bước 4: Chia tập dữ liệu (Train/Validation/Test Split)

```python
def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Chia dữ liệu thành train/validation/test
    """
    # Chia train+val và test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Chia train và validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print("Kích thước datasets:")
    print(f"  Train:      {len(X_train):5d} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):5d} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):5d} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Bước 5: Mô hình cơ sở (Baseline Model)

```python
def build_baseline(X_train, y_train, X_val, y_val):
    """
    Xây dựng baseline model đơn giản
    """
    # Baseline 1: Dự đoán bằng mean
    y_pred_mean = np.full(len(y_val), y_train.mean())
    mse_mean = mean_squared_error(y_val, y_pred_mean)
    r2_mean = r2_score(y_val, y_pred_mean)
    
    # Baseline 2: Linear Regression đơn giản
    from sklearn.linear_model import LinearRegression
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_val_scaled)
    mse_lr = mean_squared_error(y_val, y_pred_lr)
    r2_lr = r2_score(y_val, y_pred_lr)
    
    print("BASELINE MODELS")
    print("=" * 50)
    print(f"Mean Prediction:")
    print(f"  MSE: {mse_mean:.4f}")
    print(f"  R²:  {r2_mean:.4f}")
    print(f"\nLinear Regression:")
    print(f"  MSE: {mse_lr:.4f}")
    print(f"  R²:  {r2_lr:.4f}")
    
    return mse_lr  # Baseline để beat

# baseline_mse = build_baseline(X_train, y_train, X_val, y_val)
```

## Bước 6: Lựa chọn mô hình và tinh chỉnh siêu tham số

```python
def model_selection_pipeline(X_train, y_train, X_val, y_val):
    """
    So sánh nhiều models và tune hyperparameters
    """
    # Tạo preprocessing pipeline
    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    
    # Định nghĩa models và param grids
    models = {
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1, 10, 100, 1000]}
        },
        'Lasso': {
            'model': Lasso(max_iter=10000),
            'params': {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        }
    }
    
    results = {}
    
    for name, config in models.items():
        print(f"\nTuning {name}...")
        
        # Grid search với CV
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val_scaled)
        
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'mse': mse,
            'r2': r2,
            'cv_score': -grid_search.best_score_
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  CV MSE: {-grid_search.best_score_:.4f}")
        print(f"  Val MSE: {mse:.4f}")
        print(f"  Val R²: {r2:.4f}")
    
    # So sánh
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for name, res in results.items():
        print(f"{name:15s}: Val MSE = {res['mse']:.4f}, R² = {res['r2']:.4f}")
    
    # Chọn best model
    best_model_name = min(results, key=lambda x: results[x]['mse'])
    print(f"\nBest Model: {best_model_name}")
    
    return results, best_model_name, preprocessor

# results, best_name, preprocessor = model_selection_pipeline(X_train, y_train, X_val, y_val)
```

## Bước 7: Đánh giá cuối cùng trên tập kiểm tra (Test Set)

```python
def final_evaluation(best_model, preprocessor, X_test, y_test):
    """
    Đánh giá cuối cùng trên test set
    """
    X_test_scaled = preprocessor.transform(X_test)
    y_pred = best_model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print("=" * 60)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# final_evaluation(results[best_name]['model'], preprocessor, X_test, y_test)
```

## Bước 8: Diễn giải mô hình (Model Interpretation)

```python
def interpret_model(model, feature_names):
    """
    Diễn giải mô hình: feature importance
    """
    if hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_)
        title = 'Feature Coefficients (Absolute Values)'
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        title = 'Feature Importances'
    else:
        print("Model không hỗ trợ feature importance")
        return
    
    # Sort
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.show()
    
    print("\nTop 5 Most Important Features:")
    for i in range(min(5, len(importances))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]:20s}: {importances[idx]:.4f}")

# interpret_model(results[best_name]['model'], X_train.columns.tolist())
```

## Hàm tổng hợp toàn bộ quy trình

```python
def complete_ml_workflow():
    """
    Quy trình ML hoàn chỉnh từ đầu đến cuối
    """
    print("BƯỚC 1: LOAD & EXPLORE DATA")
    print("=" * 60)
    df = load_and_explore_data()
    
    print("\n\nBƯỚC 2: PREPROCESSING")
    print("=" * 60)
    X, y = preprocess_data(df)
    
    print("\n\nBƯỚC 3: FEATURE ENGINEERING")
    print("=" * 60)
    X_eng = create_features(X)
    
    print("\n\nBƯỚC 4: TRAIN/VAL/TEST SPLIT")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_eng, y)
    
    print("\n\nBƯỚC 5: BASELINE MODEL")
    print("=" * 60)
    baseline_mse = build_baseline(X_train, y_train, X_val, y_val)
    
    print("\n\nBƯỚC 6: MODEL SELECTION & TUNING")
    print("=" * 60)
    results, best_name, preprocessor = model_selection_pipeline(
        X_train, y_train, X_val, y_val
    )
    
    print("\n\nBƯỚC 7: FINAL EVALUATION")
    print("=" * 60)
    final_evaluation(results[best_name]['model'], preprocessor, X_test, y_test)
    
    print("\n\nBƯỚC 8: MODEL INTERPRETATION")
    print("=" * 60)
    interpret_model(results[best_name]['model'], X_train.columns.tolist())
    
    return results, best_name

# results, best_model_name = complete_ml_workflow()
```

## Danh sách kiểm tra thực hành tốt (Best Practices)

✅ **Hiểu dữ liệu**
- Khám phá dữ liệu (EDA) kỹ lưỡng trước khi mô hình hóa.
- Kiểm tra giá trị thiếu (missing values) và giá trị ngoại lai (outliers).
- Nắm rõ phân phối của từng đặc trưng và biến mục tiêu.

✅ **Chia tập dữ liệu**
- Chia tập train/validation/test trước khi thực hiện các phép biến đổi dựa trên thống kê dữ liệu.
- Giữ kín tập kiểm tra (test set) cho tới bước đánh giá cuối cùng.

✅ **Tiền xử lý**
- Chỉ ước lượng tham số tiền xử lý (`fit`) trên tập huấn luyện, sau đó áp dụng (`transform`) lên tập validation và test.
- Dùng `Pipeline` để tránh rò rỉ dữ liệu (data leakage).

✅ **Lựa chọn mô hình**
- Bắt đầu từ một mô hình cơ sở (baseline) đơn giản để làm mốc so sánh.
- So sánh nhiều họ mô hình khác nhau bằng kiểm định chéo (cross-validation).

✅ **Tinh chỉnh siêu tham số**
- Sử dụng `GridSearchCV` hoặc `RandomizedSearchCV`.
- Tinh chỉnh dựa trên tập validation hoặc cross-validation, không dùng tập test.

✅ **Đánh giá cuối cùng**
- Chỉ đánh giá trên tập test một lần duy nhất sau khi đã chốt mô hình.
- Báo cáo nhiều chỉ số đánh giá (MSE, RMSE, MAE, $$R^2$$) để có cái nhìn đầy đủ.

✅ **Diễn giải mô hình**
- Kiểm tra mức độ quan trọng của đặc trưng (feature importance) và phân tích phần dư (residual analysis) để hiểu hành vi của mô hình.

### Khung giải thích: tự tay tính để thấy rò rỉ dữ liệu (Data Leakage) làm méo chuẩn hóa như thế nào (nên đọc chậm)

Nhiều người nghĩ "chuẩn hóa z-score toàn bộ bảng dữ liệu trước rồi mới chia Train/Test có sao đâu?". Hãy thử ngay trên 5 con số thu nhập (triệu đồng): tập **Train** gồm $$\{10, 12, 14\}$$, còn tập **Test** tương lai bị dịch chuyển lên $$\{100, 104\}$$.

1. **Cách LÀM SAI (chuẩn hóa gộp cả 5 điểm trước khi chia):**
   - Trung bình gộp: $$\bar x_{\text{gộp}} = \frac{10 + 12 + 14 + 100 + 104}{5} = 48$$, độ lệch chuẩn $$s_{\text{gộp}} \approx 44.1$$.
   - Sau chuẩn hóa, tập Train biến thành $$\{-0.86,\, -0.82,\, -0.77\}$$, còn tập Test nằm ở $$\{+1.18,\, +1.27\}$$.
   **Lỗi nghiêm trọng:** Chỉ bằng việc chuẩn hóa, thông tin rằng "tương lai có những giá trị cực lớn quanh 100" đã **rò rỉ ngược vào tập Train** (kéo trung bình từ $$12$$ lên $$48$$), làm điểm số trên Test trông ảo giác đẹp hơn hẳn thực tế triển khai!

2. **Cách LÀM ĐÚNG (dùng `Pipeline` chỉ `fit` trên Train):**
   - Tính tham số chỉ từ Train $$\{10, 12, 14\}$$: $$\bar x_{\text{train}} = 12$$, $$s_{\text{train}} = \sqrt{\frac{4+0+4}{3}} \approx 1.63$$.
   - Biến đổi Train thành $$\{-1.22,\, 0,\, +1.22\}$$; sau đó dùng **đúng** $$\bar x_{\text{train}} = 12,\ s_{\text{train}} = 1.63$$ để biến đổi Test thành $$\{+54.0,\, +56.4\}$$!
   Lúc này, giá trị z-score khổng lồ $$+54$$ lập tức **báo động đỏ** rằng dữ liệu kiểm tra đã dịch chuyển phân phối nghiêm trọng so với lúc huấn luyện — đúng điều một kỹ sư ML cần phát hiện trước khi đưa mô hình ra thực tế!

## Video tham khảo

- [Train, Test, & Validation Sets explained](https://www.youtube.com/watch?v=Zi-0rlM4RDs) — deeplizard, tiếng Anh. **Xem trước.** Giải thích lý do cần tách riêng ba tập train, validation và test — tương ứng bước chia dữ liệu trong quy trình của bài.
- [One-Hot, Label, Target and K-Fold Target Encoding, Clearly Explained!!!](https://www.youtube.com/watch?v=589nCGeWG1w) — StatQuest with Josh Starmer, tiếng Anh. Giới thiệu mã hóa one-hot (`OneHotEncoder`) và các phương pháp mã hóa biến phân loại khác trong bước chuẩn bị đặc trưng.
- [Machine Learning Tutorial Python - 16: Hyper parameter Tuning (GridSearchCV)](https://www.youtube.com/watch?v=HdlDYng8g9s) — codebasics, tiếng Anh. Hướng dẫn sử dụng `GridSearchCV` trong scikit-learn để chọn siêu tham số, tương ứng bước tinh chỉnh trong ví dụ của bài.

## Bài tập thực hành

**Bài 1: Dự án end-to-end**
Áp dụng toàn bộ quy trình trên cho tập dữ liệu Diabetes hoặc Boston Housing và so sánh ít nhất 3 mô hình khác nhau.

**Bài 2: Kỹ thuật đặc trưng (Feature Engineering)**
Tạo thêm ít nhất 5 đặc trưng mới cho tập dữ liệu California Housing và đánh giá xem chúng có cải thiện hiệu suất mô hình hay không.

**Bài 3: Tự động hóa bằng Pipeline**
Xây dựng một `Pipeline` hoàn chỉnh trong scikit-learn kết hợp tiền xử lý, tạo đặc trưng và huấn luyện mô hình.
