'''
	Implementação da rede neural Perceptron

	w = w + N * (d(k) - y) * x(k)
'''

import random, copy

class Perceptron:

	def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):

		self.amostras = amostras # todas as amostras
		self.saidas = saidas # saídas respectivas de cada amostra
		self.taxa_aprendizado = taxa_aprendizado # taxa de aprendizado (entre 0 e 1)
		self.epocas = epocas # número de épocas
		self.limiar = limiar # limiar
		self.num_amostras = len(amostras) # quantidade de amostras
		self.num_amostra = len(amostras[0]) # quantidade de elementos por amostra
		self.pesos = [] # vetor de pesos

	def treinar(self):
		
		for amostra in self.amostras:
			amostra.insert(0, -1)

		for i in range(self.num_amostra):
			self.pesos.append(random.random())

		self.pesos.insert(0, self.limiar)

		num_epocas = 0

		while True:

			erro = False # o erro inicialmente inexiste

			for i in range(self.num_amostras):

				u = 0

				'''
					realiza o somatório, o limite (self.amostra + 1)
					é porque foi inserido o -1 para cada amostra
				'''
				for j in range(self.num_amostra + 1):
					u += self.pesos[j] * self.amostras[i][j]

				y = self.sinal(u)

				if y != self.saidas[i]:

					erro_aux = self.saidas[i] - y

					for j in range(self.num_amostra + 1):
						self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]

					erro = True 

			num_epocas += 1

			if num_epocas > self.epocas or not erro:
				break


	# função utilizada para testar a rede
	# recebe uma amostra a ser classificada e os nomes das classes
	# utiliza a função sinal, se é -1 então é classe1, senão é classe2
	def testar(self, amostra, classe1, classe2):

		amostra.insert(0, -1)

		u = 0
		for i in range(self.num_amostra + 1):
			u += self.pesos[i] * amostra[i]

		y = self.sinal(u)

		if y == -1:
			print('A amostra pertence a classe %s' % classe1)
		else:
			print('A amostra pertence a classe %s' % classe2)


	def sinal(self, u):
		return 1 if u >= 0 else -1

amostras = [[5.1, 3.5, 1.4, 0.2],
			[4.9,3.0,1.4,0.2],
			[4.7,3.2,1.3,0.2],
			[4.6,3.1,1.5,0.2],
			[5.0,3.6,1.4,0.2],
			[7.0,3.2,4.7,1.4],
			[6.4,3.2,4.5,1.5],
			[6.9,3.1,4.9,1.5],
			[5.5,2.3,4.0,1.3],
			[6.5,2.8,4.6,1.5]]

saidas = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

rede = Perceptron(amostras=amostras, saidas=saidas,	
						taxa_aprendizado=0.1, epocas=1000)

rede.treinar()

testes = [[4.9,2.4,3.3,1.0], [6.3,3.3,4.7,1.6], [4.6,3.2,1.4,0.2]]

for teste in testes:
	rede.testar(teste, 'Iris-setosa', 'Iris-versicolor')
