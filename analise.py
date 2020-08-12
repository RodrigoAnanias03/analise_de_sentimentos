#Importando os módulos necessários.
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
import tweepy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as pyplot

# Armazenando os dados da API:
#Obs.: Os dados abaixo são privados. São identificadores individuais de cada aplicação e do desenvolvedor que a está
#utilizando. Os dados podem ser obtidos a partir do site https://developer.twitter.com
# Tokens de acesso
from textblob import TextBlob
from unidecode import unidecode

consumerKey = ""
consumerSecret = ""
accessToken = ""
accessTokenSecret = ""
# Autenticando
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)
# Passando os dados da API para a aplicação python:
auth = OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)



#Criando o arquivo para armazenar os tweets
arquivo = open('BASEDETESTE.txt', 'a', encoding='utf8')


#Criando a função que coleta os tweets. O caractere "|" foi adicionado ao final de cada tweet, pois é necessário
#um identificador do término de cada tweet para posterior separação, já que quando o algoritmo lê as bases, coloca
#todos o conteúdo em uma única string.

class MyStreamListener(tweepy.StreamListener):
        print("buscando os tweets")
        for tweet in tweepy.Cursor(api.search, q=['#covid-19', '#coronavirus', '#covid19'],
                            since='2020-07-6', until="2020-07-10", lang="pt").items(10000):
            # Texto do Tweet
            textPT = unidecode(tweet.text)
            # Traduzindo para o Inglês
            # Exibindo...
            print("[PT-BR] " + textPT)
            arquivo.write('{}{}'.format(textPT, ' |'))

#Chamando a API e a função de coleta
MyStreamListener = MyStreamListener()

#A partir daqui a base de treino do classificador está criada. As linhas 63 a 76 referem-se à criação do modelo
#classificador.
print("lendo base de treino")
dataset = pd.read_csv('BASEDETREINO.csv') #Lendo a base de dados
dataset.count() #Contando o numero de linhas

tweets = dataset['Text'].values #Armazenando todos os tweets
classes = dataset['Classificacao'].values #Armazenando todas as classificações

#Configurando o vetorizador para separar palavra por palavra
vectorizer = CountVectorizer(analyzer="word")
#Aplicando o vetorizador
freq_tweets = vectorizer.fit_transform(tweets)
#Definindo o método
modelo = MultinomialNB() #Para essa metodologia é utilizado o Naive Bayes Multinomial. Para outros, consulte a documentação do scikit-learn.
#Aplicando o método, com os tweets vetorizados e as classificações
modelo.fit(freq_tweets,classes)

#A partir daqui o modelo classificador já está pronto. Para criar a base que deverá ser rotulada pelo classificador
#(chamamos aqui de BASEDETESTE), basta repetir os mesmos passos da criação da base de treino.

arquivo = open ('BASEDETESTE.txt', 'r', encoding='utf8') #Abrindo a base de teste
planificar = arquivo.read() #Armazenando o conteúdo do arquivo em uma string
#lower = lower.lower(' \n', '\n') #Padronizando a quebra de linha do documento.
planificar = planificar.replace('\n','') #Planificando a string
testes = planificar.split('|') #Separando a string em uma lista, passando como parâmetro de separação o caractere |

freq_testes = vectorizer.transform(testes) #Aplicando o modelo

#Contando as ocorrências de cada classe
vetor = modelo.predict(freq_testes)
positivo = 0
negativo = 0
neutro = 0
for i in range(1, len(vetor)-1):
    if vetor[i] == 'Positivo':
        positivo +=1
    elif vetor[i] == 'Negativo':
        negativo +=1
    else:
        neutro +=1

#Plotando os gráficos
x_list = [positivo, negativo, neutro]
labels_list = ['Positivo', 'Negativo', 'Neutro']
pyplot.axis('equal')
pyplot.pie(x_list, labels=labels_list, colors=['Green','red','silver'], autopct='%1.2f%%')
pyplot.title('Análise de Sentimentos durante a pandemia de Covid-19 no Brasil')
pyplot.show()