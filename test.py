import tensorflow as tf
import numpy as np
import pdb

x = tf.Variable(np.arange(24).reshape([-1,3,2]),dtype=tf.float32)
y = tf.Variable(np.arange(24,48).reshape([-1,3,2]),dtype=tf.float32)
norms_x = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
norms_y = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)

#indices = tf.Variable(np.random.randint(3,size=4))#.reshape([-1,1]))
#indices = tf.multinomial(np.ones([4,3]), 1)
#pdb.set_trace()
#ind = tf.stack([np.arange(4),indices],axis=-1)


dotprod = tf.tensordot(x, tf.transpose(y), [[-1],[0]])
norm_nk = tf.tensordot(norms_x,tf.ones(tf.shape(tf.transpose(norms_x))),[[-1],[0]])
norm_lm = tf.tensordot(tf.ones(tf.shape(norms_x)),tf.transpose(norms_x),[[-1],[0]])


#inp = np.arange(24).reshape([-1,3,2])
#ind = np.random.randint(3,size=4)
#pdb.set_trace()
#idx = tf.concat((np.arange(4).reshape([-1,1]).astype('int64'),indices,-tf.ones([4,1],dtype=tf.int64)),axis=-1)
#test = inputs[np.arange(4),indices]
#test = tf.gather(inputs,indices,axis=1)
#test = tf.gather_nd(inputs,ind)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

x_ = x.eval()
y_ = y.eval()
nx = norms_x.eval()
ny = norms_y.eval()
pdb.set_trace()
