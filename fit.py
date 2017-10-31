import tensorflow as tf
from model import Model
import sys

m = Model()

inputs = sys.argv[2].split('-')
features = [e.split(',') for e in inputs]

labels = m.fit(sys.argv[1], features)
sys.stdout.write(','.join(str(e[0]) for e in labels))
