import tensorflow as tf
import numpy as np
import pdb
import math

#x = tf.Variable(np.arange(60).reshape([-1,3,2]),dtype=tf.float32)
#y = tf.Variable(np.arange(24,48).reshape([-1,3,2]),dtype=tf.float32)
#norms_x = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
#norms_y = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
#logits = tf.ones([10,3])
#idx = tf.Variable(tf.multinomial(logits, 1),dtype=tf.int64)
#mixture_idx = 0.0
#idx = tf.reshape(idx,[-1])
#mixture = tf.stack([tf.range(10,dtype=tf.int64),idx],axis=-1)
#encoded = tf.gather_nd(x,mixture)
a = 7.5
b = 7
ca = math.ceil(a)
cb = math.ceil(b)
pdb.set_trace()


# x = np.arange(60).reshape([-1,3,2]).astype(np.float32)
# samples_per_mixture = int(10 / 3)
# class_i = np.repeat(np.arange(3),samples_per_mixture,axis=0)
# mixture = np.zeros([10,],dtype='int32')
# mixture[(10 % 3):] = class_i
# test = x[np.arange(10),mixture]
# pdb.set_trace()


#indices = tf.Variable(np.random.randint(3,size=4))#.reshape([-1,1]))
#indices = tf.multinomial(np.ones([4,3]), 1)
#pdb.set_trace()
#ind = tf.stack([np.arange(4),indices],axis=-1)



#test = int(1.75)
#pdb.set_trace()
# for i in range(1):
#     print(i)
#     pdb.set_trace()
#
# dotprod = tf.tensordot(x, tf.transpose(y), [[-1],[0]])
# norm_nk = tf.tensordot(norms_x,tf.ones(tf.shape(tf.transpose(norms_x))),[[-1],[0]])
# norm_lm = tf.tensordot(tf.ones(tf.shape(norms_x)),tf.transpose(norms_x),[[-1],[0]])


#inp = np.arange(24).reshape([-1,3,2])
#ind = np.random.randint(3,size=4)
#pdb.set_trace()
#idx = tf.concat((np.arange(4).reshape([-1,1]).astype('int64'),indices,-tf.ones([4,1],dtype=tf.int64)),axis=-1)
#test = inputs[np.arange(4),indices]
#test = tf.gather(inputs,indices,axis=1)
#test = tf.gather_nd(inputs,ind)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

idx_ = idx.eval()
mixture_ = mixture.eval()
encoded_ = encoded.eval()
pdb.set_trace()
