import tensorflow as tf
from model import Model
import sys

m = Model()

features = sys.argv[2].split(',')

labels = m.fit(sys.argv[1], [features])
sys.stdout.write(','.join(str(e) for e in labels[0]))
