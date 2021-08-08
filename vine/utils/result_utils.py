import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy as np

def draw_blocking_prob(x, labels, colors, x_label, y_label, title='', save=True, name=None, scale=1):
  # sns.set()
  fig, ax = plt.subplots()
  fig.set_size_inches(10, 7)

  for i in range(len(x)):
    ax.plot(np.array(range(len(x[i])))[::scale], x[i][::scale],  color=colors[i], label=labels[i])
  ax.set_ylabel(y_label)
  ax.set_xlabel(x_label)
  ax.grid()
  legend = ax.legend(loc='upper left', shadow=True)
  if title != '':
    ax.set_title(title)
  ax.legend(loc="bset")

  if save:
    image_name = name if name is not None else time.strftime('%a,%d-%b-%Y-%I:%M:%S')
    fig.savefig(f'./results/compare/{image_name}.png')
    import tikzplotlib
    tikzplotlib.save(f'./results/compare/{image_name}.tex')

  return ax, fig


def save_data(data, name):
  path = f'./results/{name}.dat'
  np.array(data).tofile(path)


def draw_bars(labels, values, title, save=True, name=None):
  # sns.set()
  fig, ax = plt.subplots()
  sns.barplot(labels, values, ax=ax)
  ax.set_title(title)

  if save:
    image_name = name if name is not None else time.strftime('%a,%d-%b-%Y-%I:%M:%S')
    fig.savefig(f'./results/compare/{image_name}.png')

    csv_str = 'Name,Value'
    for i in range(len(values)):
      csv_str += f'\n{labels[i]},{values[i]}'
    file1 = open(f'./results/compare/{image_name}.csv',"w")#write mode 
    file1.write(csv_str) 
    file1.close() 

    import tikzplotlib
    tikzplotlib.save(f'./results/compare/{image_name}.tex') 
  return ax, fig