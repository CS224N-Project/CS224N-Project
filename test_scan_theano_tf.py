import numpy as np

import theano
import theano.tensor as T

import tensorflow as tf

rng = np.random

from collections import OrderedDict

# train_generator = theano.function(
#         inputs = [ self_x, self_y ],
#         outputs = [ self_generator_obj, self_generator_loss, \
#                         self_generator_sparsity_cost, self_z, gnorm_g, gnorm_e ],
#         givens = {
#             self_z : self_generator_z_pred
#         },
#         updates = updates_g.items() + updates_e.items(),
#     )

# ...

# cost, loss, sparsity_cost, bz, gl2_g, gl2_e = train_generator(bx, by)


print '\n ***** theano scan - TEST *****'

SIZE = 3

# theano
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

# tensorflow (dtypes and shapes required)
X_tf = tf.placeholder(dtype=tf.float32, shape=(SIZE,SIZE))
W_tf = tf.placeholder(dtype=tf.float32, shape=(SIZE,SIZE))
b_sym_tf = tf.placeholder(dtype=tf.float32, shape=(SIZE,))

# theano
def compute_one_element(v):
	#print 'compute_one_element' # <-- prints at ** compile time ** when execution thread hits theano.scan below
	# example hack to print variable value at runtime (not at compile time):
	compute_graph = (T.tanh(T.dot(v, W) + b_sym)) + (1e-11 * theano.printing.Print('v is: ')(v))
	#theano.printing.debugprint(compute_graph) # <-- prints at ** compile time **
	return compute_graph

# tensorflow
def compute_one_element_tf(prev, curr):
	#print 'compute_one_element_tf' # <-- prints at ** compile time ** when execution thread hits tf.scan below
	# example hack to print variable value at runtime (not at compile time):
	compute_graph = (tf.tanh(tf.matmul(tf.reshape(curr, [1,SIZE]), W_tf) + b_sym_tf)) + tf.Print(0.0, data=[curr])

	return compute_graph


# theano.scan calls compute_one_element() above for each ** row ** of X (passed in as variable v)
results, _ = theano.scan(
		fn = compute_one_element,
		sequences = X
	)

# tf.scan also seems to call compute_one_element_tf() above for each ** row ** of X (passed in as variable curr)
scan_func_tf = tf.scan(
	fn = compute_one_element_tf,
	elems = X_tf,
	initializer = tf.zeros(shape=(1,SIZE), dtype=tf.float32)
	)

# alternate theano approach to creating print functions to call later:
print_X = theano.printing.Print('X is: ')(X)
print_W = theano.printing.Print('W is: ')(W)
print_b_sym = theano.printing.Print('b_sym is: ')(b_sym)
print_vals = theano.function(
		inputs = [ X, W, b_sym ],
		outputs = [ print_X, print_W, print_b_sym ]
	) 


shared_var = theano.shared(np.ones((SIZE), dtype=np.float32))

compute_elementwise = theano.function(
        inputs = [ X, W, b_sym ],
        outputs = [ results, shared_var, print_X, print_W, print_b_sym ],
        updates = {shared_var: shared_var + b_sym}
    )

### ACTUAL DATA AND COMPUTATION BELOW ###

# test values
x = np.eye(SIZE, dtype=np.float32)
w = np.ones((SIZE, SIZE), dtype=np.float32)
b = np.ones((SIZE), dtype=np.float32)
b[SIZE-1] = 2

print '\n theano - shared_var before updates:'
print shared_var.eval()

print '\n theano - compute_elementwise:'
print(compute_elementwise(x, w, b)[0])

print '\n theano - shared_var after updates:'
print shared_var.eval()

print '\n theano - print_vals:'
print_vals(x, w, b)

print '\n tensorflow:'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ''
    print(sess.run(scan_func_tf, feed_dict={ X_tf: x, W_tf: w, b_sym_tf: b }))
    print ''

print '\n numpy:'
print(np.tanh(x.dot(w) + b))

print '\n addl test - theano:'

x = theano.shared(np.asarray([7.1,2.2,3.4], dtype = np.float32))

v = T.vector("v")
def fv(v):
    res,_ = theano.scan(lambda x: x ** 2, v)
    return T.sum(res)

def f(i):
    return fv(x[i:i+2])

outs,_ = theano.scan(
    f, 
    T.arange(2)
    )

fn = theano.function(
    [],
    outs,
    )

print fn()

print '\n addl test - tensorflow:'



# TODO ABOVE - test with "updates" parameter and in corresponding tensorflow version:
# train_generator = theano.function(
#         inputs = [ self.x, self.y ],
#         outputs = [ self.generator.obj, self.generator.loss, \
#                         self.generator.sparsity_cost, self.z, gnorm_g, gnorm_e ],
#         updates = updates_g.items() + updates_e.items()
#     )
# NOTE: theano updates should be a list (or dictionary) of key value pairs where the key is a shared variable
# and the value is a symbolic expression describing how to update the corresponding shared variable.

# TODO ABOVE - test with "outputs_info" parameter and in corresponding tensorflow parameter (initializer):
# h, _ = theano.scan(
#             fn = self.forward,
#             sequences = x,
#             outputs_info = [ h0 ]
#         )

# http://deeplearning.net/software/theano/library/scan.html#theano.scan
# outputs_info is the list of Theano variables or dictionaries describing the initial state 
# of the outputs computed recurrently.
# http://stackoverflow.com/questions/34079625/theano-scan-how-outputs-info-with-placeholder-feed-as-input
# For each value in outputs_info, if it is None then you are saying that the value is not recurrent 
# (outputs in this position are not made available to later iterations). A non-None outputs_info value
# indicates that the step output in that position is recurrent and the value provided is the initial value 
# to be be passed to the step function in the first step; subsequent steps receive the output from the previous step.



print '\n addl test 2 - theano:'
#http://ir.hit.edu.cn/~jguo/docs/notes/a_simple_tutorial_on_theano.pdf

N = 400 # number of samples
feats = 784 # dimensionality of features
D = (rng.randn(N, feats).astype(np.float32), rng.randint(size=N, low=0, high=2).astype(np.float32))

training_steps = 10000
x = T.matrix('x')
y = T.vector('y')
w = theano.shared(rng.randn(784), name='w')
b = theano.shared(0., name='b')
p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b))
prediction = (p_1 > 0.5)
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
cost = xent.mean() + 0.01 * (w**2).sum()
gw, gb = T.grad(cost, [w, b])

updates_dict = {w : w-0.1*gw, b : b-0.1*gb} #using standard dict theano.function throws a warning
# OrederedDict works:
updates_ordereddict = OrderedDict()
updates_ordereddict[w] = w-0.1*gw
updates_ordereddict[b] = b-0.1*gb
# list of tuples also works:
updates_list = [(w, w-0.1*gw), (b, b-0.1*gb)]


# Compile
train = theano.function(
	inputs = [x, y],
	outputs = [prediction, xent],
	updates = updates_list
	)

predict = theano.function(inputs = [x], outputs = prediction)

# Train
for i in range(training_steps):
	pred, err = train(D[0], D[1])

print 'Final model:'
print w.get_value(), b.get_value()
print 'target values for D: ', D[1]
print 'predictions on D: ', predict(D[0])

print '\n addl test 2 - tensorflow:'




