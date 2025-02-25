import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Fator de conversão de pés quadrados para metros quadrados
SQFT_TO_M2 = 0.092903

# 1. Carregamento dos Dados
df_user = pd.read_csv('user.csv')
df_transaction = pd.read_csv('transactions.csv')
df_property = pd.read_csv('property_data.csv')

# Converter 'size_sqft' para 'size_m2' em df_property
df_property['size_m2'] = df_property['size_sqft'] * SQFT_TO_M2

# 2. Integração dos Dados
# Mescla df_transaction e df_property utilizando 'property_id' como chave
df_merged = pd.merge(df_transaction, df_property, on='property_id', how='inner')

# 3. Engenharia de Atributos

# Realiza one-hot encoding na coluna 'location' (drop_first para evitar multicolinearidade)
df_merged = pd.get_dummies(df_merged, columns=['location'], drop_first=True)

# Calcular o preço por metro quadrado (opcional)
df_merged['price_per_m2'] = df_merged['sale_price'] / df_merged['size_m2']

# Atualizar as features: removemos 'property_age' e adicionamos as variáveis dummies da localização
dummy_columns = [col for col in df_merged.columns if col.startswith('location_')]
features = ['size_m2', 'bedrooms', 'bathrooms', 'listing_price'] + dummy_columns
target = 'sale_price'

X = df_merged[features]
y = df_merged[target]

# 4. Divisão dos Dados e Treinamento do Modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliação do modelo utilizando RMSE
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write("RMSE do modelo:", rmse)

# 5. Criação da Interface com Streamlit
st.title("Previsão de Preço de Venda de Imóveis")

st.sidebar.header("Dados do Imóvel para Previsão")
# Entrada para tamanho em m², número de quartos, banheiros e preço de listagem
size_m2 = st.sidebar.number_input("Tamanho (m²)", min_value=50, max_value=1000, value=140, step=10)
bedrooms = st.sidebar.number_input("Número de Quartos", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Número de Banheiros", min_value=1, max_value=10, value=2)
listing_price = st.sidebar.number_input("Preço de Listagem", min_value=50000.0, max_value=1000000.0, value=300000.0, step=1000.0)

# Seleção da localização do imóvel
unique_locations = sorted(df_property['location'].unique())
selected_location = st.sidebar.selectbox("Localização", unique_locations)

# Definindo a categoria base (aquela dropada durante o one-hot encoding)
baseline_location = unique_locations[0]

if st.sidebar.button("Prever Preço de Venda"):
    # Criar dummies para a localização conforme utilizado no treinamento
    location_dummies = {col: 0 for col in dummy_columns}
    if selected_location != baseline_location:
        dummy_col = f"location_{selected_location}"
        if dummy_col in location_dummies:
            location_dummies[dummy_col] = 1

    # Organizar os dados de entrada em um DataFrame
    input_dict = {
        'size_m2': [size_m2],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'listing_price': [listing_price]
    }
    # Adicionar as variáveis dummy de localização
    for col in dummy_columns:
        input_dict[col] = [location_dummies.get(col, 0)]
        
    input_data = pd.DataFrame(input_dict)
    
    # Realiza a previsão
    predicted_price = model.predict(input_data)[0]
    st.subheader("Resultado da Previsão")
    st.write(f"O preço de venda previsto é: $ {predicted_price:,.2f}")
