import tensorflow as tf
from model import Model
import sys

m = Model()

features = sys.argv[3].split(',')
labels = sys.argv[4].split(',')

m.train(sys.argv[1], sys.argv[2], [features], [labels])
