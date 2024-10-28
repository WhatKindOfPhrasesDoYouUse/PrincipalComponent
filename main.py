import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def load_data():
    return pd.read_csv('close_prices.csv')
    
def load_djia():
    return pd.read_csv('djia_index.csv')

def pca_learn():
    data = load_data()
    X = data.iloc[:, 1:] # убрал столбец с датой 
    model = PCA(n_components=10)
    model.fit(X)

    n_components_90_var = np.where(np.cumsum(model.explained_variance_ratio_) >= 0.9)[0][0] + 1
    print('Количество компонентов, объясняющих 90% дисперсии: ', n_components_90_var)

    X_pca = model.transform(X)
    first_component = X_pca[:, 0]
    return model, first_component

def analyze_correlation():
    model, first_component = pca_learn()
    djia = load_djia()
    
    correlation = np.corrcoef(first_component, djia['^DJI'])[0, 1]
    print("Корреляция Пирсона между первой компонентой и индексом Доу-Джонса: ", correlation)
    
    max_weight_idx = np.argmax(np.abs(model.components_[0]))
    company_with_max_weight = load_data().columns[max_weight_idx + 1]  # +1 из-за столбца даты
    print("Компания с наибольшим весом в первой компоненте: ", company_with_max_weight)
    
def main():
    pca_learn()
    analyze_correlation()

if __name__ == '__main__':
    main()