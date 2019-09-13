# Load libraries
import pandas as pd
import numpy as np
import re
import nltk
# O Seguinte comando é executado para adquirir o pacote utilizado para
# criar o vetor utilizado no algorítmo de machine learning
nltk.download('stopwords')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# O Seguinte comando é utilizado para executar funções do matplotlib no
# jupyter notebook
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

############## Pré Processamento ###########################

# Carregando arquivo para compor dataset de treino
fields = ['id', 'label', 'tweet']
dataset = pd.read_csv("./twitter-sentiment-analysis-hatred-speech/train.csv", names=fields)
# dataset = dataset.head(2000)
# print(dataset)

# Reformulando forma de printar as representações visuais
plot_size = plt.rcParams["figure.figsize"] # Alteração dos dados da figura
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size
dataset.label.value_counts().plot(kind='pie', autopct='%1.0f%%') # Plotagem da representação visual no estilo "Torta"
# plt.show()

############ Escolha das Features #######################

# O Conjunto de Features consistirão de, somente, os tweets em forma textual,
# e, para isso, é preciso que os dados apresentados sejam tratados para o
# treinamento do modelo que classifique o discurso de ódio
# O Conjunto de Labels que marcarão o conteúdo do tweet será a própria coluna de label

features = dataset.iloc[:, 2].values
labels = dataset.iloc[:, 0].values

features_processed = []

for sentence in range(0, len(features)):
    # Remoção de todos os caracteres especiais e substituição desses caracteres por espaços
    feature_processed = re.sub(r'\W', ' ', str(features[sentence]))

    # Remoção de todos os caracteres sozinhos como resultado da remoção dos caracteres especiais
    feature_processed= re.sub(r'\s+[a-zA-Z]\s+', ' ', feature_processed)

    # Remoção de caracteres sozinhos do começo do tweet
    feature_processed = re.sub(r'\^[a-zA-Z]\s+', ' ', feature_processed)

    # Substitui espaços múltiplos em decorrência das substituições anteriores por um único espaço
    feature_processed = re.sub(r'\s+', ' ', feature_processed, flags=re.I)

    # Remoção de caracteres com prefixo em b, que significa que são caracteres em bytes
    feature_processed = re.sub(r'^b\s+', '', feature_processed)

    # Conversão de todos os caracteres para letras minúsculas
    feature_processed = feature_processed.lower()

    features_processed.append(feature_processed)

# Processo de conversão das features processadas para vetores de TF-IDF, que cria um
# vetor com a frequência das palavras, considerando que as palavras que ocorrem menos em todos
# os documentos, mas que ocorrem mais em um documento contribuem para sua classificação, sendo
# assim, a combinação de dois termos, um Termo de Frequência, representando a razão entre a
# frequência de uma palvra no documento e o total de palavras no documento. Enquanto o outro termo
# é o IDF, Documento Inverso de Frequência, tido como o Logarítmo da razão do número total de documentos
# pelo número de documentos contendo a palavra

################# Criação do Modelo de Machine Learning ############################

# A Função a seguir possui a seguinte construção
# max_features - Número de palavra mais frequentes para criar o vetor de Bag of Words
# min_df - Indica o número mínimo de documentos em que a palavra deve aparecer para ser incluída
# max_df - Especifica a porcentagem máxima de documentos nos quais aquela palavra deve aparecer para ser incluída, evitando palavras muito comuns

vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
features_processed = vectorizer.fit_transform(features_processed).toarray()
print('features_processed')

# Como orientado, foi utilizado somente o dataset de treino para compor o exercício, então
# esse dataset foi dividido entre um conjunto de treino e um conjunto de teste, com 25% do conjunto para
# treinamento e 75% de teste
# Assim, os elementos marcados como x atribuídos no retorno da próxima função atuam como as features
# e os elementos em y, como as labels

x_train, x_test, y_train, y_test = train_test_split(features_processed, labels, test_size=0.25, random_state=0)
print('splitted sets')

##################### Treinamento ######################################

# Para treinar o modelo, foi escolhido o algorítmo de floresta aleatória (Random Forest Algorithm) pois sua versatilidade
# ao trabalhar com dados não normalizados é de interesse, ao utilizar como base as árvores de decisão para compor o algorítmo,
# permitindo uma realimentação de dados não enviezada
# Assim, o módulo sklearn.ensemble contém um algorítmo de RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(x_train, y_train)
print('text classified')

# Com o modelo treinado, o método 'predict' permite realizar previsões acerca do conjunto de teste utilizado, utilizando, assim,
# o conjunto x para tal

predictions = text_classifier.predict(x_test)
print('predicitions made')

######################## Escolha da Métrica ########################################

# Dentre as possibilidades de escolha de métrica, a matriz de confusão apresenta os valores previstos e os valores, de fato, obtidos
# permitindo uma identificação visual dos falso positivos, falso negativos e valores corretamento correlacionados,

print(confusion_matrix(y_test,predictions))

# A Classificação reportada, utilizando o fator F1 pode ser visualizada através da saída do algorítmo abaixo
# contendo uma medição da acurácia do teste, analisando a precisão do resultado obtido
print(classification_report(y_test,predictions))

# A Classificação por precisão pode ser observada pelos resultados abaixo
print(accuracy_score(y_test, predictions))
