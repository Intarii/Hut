import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

class Helplot:
  """
  TensorFlow Model Training History Plot
  
  >>> history = model.fit(train, ...)
  >>> plot_hist = Helplot(history)
  >>> plot_hist.Relplot
  >>> plot_hist.Falplot
  """
  def __init__(self, hist, loop):
    self.hist = hist
    self.loop = loop

  def plotlabel(title, xlabel, ylabel, legend=False):
    """
    Labeler
    """
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(legend)
  
  @property
  def Relplot(self):
    """
    Plot on Result
    """
    plt.plot(self.loop)
    if 'val_accuracy' in self.hist:
      plt.plot(self.loop, self.val_accuracy, label='Val Accuracy')
    self.plotlabel()

  @property
  def Falplot(self):
    """
    Plot on Fail Rate
    """
    if 'val_loss' in self.hist:
      plt.plot(self.loop, self.val_loss, label='Val Loss')
    self.plotlabel()