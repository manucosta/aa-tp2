import numpy as np
import matplotlib.pyplot as plt
import sys

def grid_search(rotulos, resultados):
  # Example data
  y_pos = np.arange(len(rotulos))
  performance = resultados

  fig = plt.figure(figsize=(13,20))
  ax = fig.add_subplot(111)
  ax.barh(y_pos, performance, align='center', alpha=0.4)
  plt.yticks(y_pos, rotulos)
  plt.xlabel('Performance')
  plt.xlim(0.4, 0.8)
  plt.title('Tasa de victorias contra Random')
  plt.tight_layout()
  #plt.show()
  plt.savefig('../graficos/grid_search.pdf')

def parser(file):
  rotulos = []
  resultados = []
  fo = open(file, 'r')
  lines = fo.readlines()
  fo.close()
  lines = map(lambda x: x.rstrip(), lines)
  for i in xrange(len(lines)):
    if i%2 == 0:
      rotulos.append(lines[i])
    else:
      lista = lines[i].split(' ')
      resultados.append(lista[-1])
  return rotulos, resultados

rotulos, resultados = parser(sys.argv[1])
resultados, rotulos = zip(*sorted(zip(resultados, rotulos)))
grid_search(rotulos, resultados)