#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import os
from env import *
import pickle

class Qnetwork():
    def __init__(self, is_training):
        global num_steps
        global hidden_size
        global vocab_size
        global keep_prob
        global action_num

        #inputs: features, mask
        self.text_input = tf.placeholder(tf.int32, [None, 4*num_steps], name="text_input")
        self.query,self.candidate1,self.candidate2,self.candidate3=tf.split(self.text_input, num_or_size_splits=4, axis=1)
        self.text_mask = tf.placeholder(tf.int32, [None, 4], name="text_mask")
        self.query_mask,self.candidate1_mask,self.candidate2_mask,self.candidate3_mask=tf.split(self.text_mask, num_or_size_splits=4, axis=1)
        self.query_mask=tf.reshape(self.query_mask,[-1])
        self.candidate1_mask=tf.reshape(self.candidate1_mask,[-1])
        self.candidate2_mask=tf.reshape(self.candidate2_mask,[-1])
        self.candidate3_mask=tf.reshape(self.candidate3_mask,[-1])

        #word embedding layer
        with tf.device("/cpu:0"):
            self.embedding=embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
            # batch_size * num_steps* embedding_size
            query_inputs = tf.nn.embedding_lookup(embedding, self.query)
            candidate1_inputs = tf.nn.embedding_lookup(embedding, self.candidate1)
            candidate2_inputs = tf.nn.embedding_lookup(embedding, self.candidate2)
            candidate3_inputs = tf.nn.embedding_lookup(embedding, self.candidate3)
            #add dropout to input units
            if is_training and keep_prob < 1:
                query_inputs = tf.nn.dropout(query_inputs, keep_prob)
                candidate1_inputs = tf.nn.dropout(candidate1_inputs, keep_prob)
                candidate2_inputs = tf.nn.dropout(candidate2_inputs, keep_prob)
                candidate3_inputs = tf.nn.dropout(candidate3_inputs, keep_prob)

        #add LSTM cell and dropout nodes
        with tf.variable_scope('forward'):
            fw_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
            if is_training and keep_prob < 1:
                fw_lstm = tf.contrib.rnn.DropoutWrapper(fw_lstm, output_keep_prob=keep_prob)

        with tf.variable_scope('backward'):
            bw_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
            if is_training and keep_prob < 1:
                bw_lstm = tf.contrib.rnn.DropoutWrapper(bw_lstm, output_keep_prob=keep_prob)

        #bidirectional rnn
        query_output=tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=query_inputs, sequence_length=self.query_mask, dtype=tf.float32)
        candidate1_output=tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=candidate1_inputs, sequence_length=self.candidate1_mask, dtype=tf.float32)
        candidate2_output=tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=candidate2_inputs, sequence_length=self.candidate2_mask, dtype=tf.float32)
        candidate3_output=tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=candidate3_inputs, sequence_length=self.candidate3_mask, dtype=tf.float32)
        
        #batch_size * num_step * (hidden_size, hidden_size)
        self.query_output=query_output=tf.concat(query_output[0], 2)
        self.candidate1_output=candidate1_output=tf.concat(candidate1_output[0], 2)
        self.candidate2_output=candidate2_output=tf.concat(candidate2_output[0], 2)
        self.candidate3_output=candidate3_output=tf.concat(candidate3_output[0], 2)
        #final sentence embedding.  batch_size * (2 * hidden_size)
        self.query_output=query_output=tf.reduce_mean(query_output, axis=1)
        self.candidate1_output=candidate1_output=tf.reduce_mean(candidate1_output, axis=1)
        self.candidate2_output=candidate2_output=tf.reduce_mean(candidate2_output, axis=1)
        self.candidate3_output=candidate3_output=tf.reduce_mean(candidate3_output, axis=1)
        self.features=features=tf.concat([query_output, candidate1_output, candidate2_output, candidate3_output], axis=1)

        transform_w = tf.get_variable("transform_w", [8*hidden_size, action_num])
        transform_b = tf.get_variable("transform_b", [action_num])
        self.Qout=Qout = tf.matmul(features, transform_w) + transform_b
 
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        if is_training:
            self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,action_num,dtype=tf.float32)
            
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
            
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.01)
            self.updateModel = self.trainer.minimize(self.loss)
            #get gradients by debugging
            tvar = tf.trainable_variables()
            self.grads=tf.gradients(self.loss, tvar)


class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) > self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
        
    def sample(self,size):
        global action_num
        return np.array(random.sample(self.buffer,size))

#convert text to number and extend to num_steps
def processState(states, env):
    global num_steps
    dictionary=env.dictionary
    new_states=[]
    lengths=[]
    is_query=True
    for context in states:
        #convert word to number
        state=[0]*(num_steps)
        rev_context=[element for element in context]
        #add descritpion
        #if is_query:
        #    is_query=False
        #    rev_context+=(env.description.get(env.concept,[])[:100])
        rev_context.reverse()
        word_to_num=[dictionary.get(word, 0) for word in rev_context]
        state[:len(word_to_num)]=word_to_num
        new_states+=state
        #record length
        lengths.append(len(rev_context))
    return np.array(new_states), np.array(lengths)

#append all update operators
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

#update the parameters
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

#test mdoel performance
def test_model(model, game, l, flag):
    global query_max
    avg_accuracy=0
    avg_recall=0
    avg_map=0
    avg_p10=0
    count=0
    for concept in l:
        #Reset environment and get first new observation
        state = game.reset(concept)
        processed_state, state_lengths = processState(state, game)
        
        #record how long the query is for this episode (to prevent overflow length query)
        query_size=1
        while query_size<query_max:
            action = sess.run(mainQN_test.predict,feed_dict={mainQN_test.text_input: [processed_state], mainQN_test.text_mask:[state_lengths]})[0]
            print("ACTION: "+str(action))
            if action<3:
                query_size+=1
            state1,reward,stop = game.step(action)
            processed_state1, state1_lengths = processState(state1, game)
              
            state = state1
            processed_state=processed_state1
            state_lengths=state1_lengths      
            if stop == True:
                break
        accuracy=game.accuracy(state[0], 5)
        p10=game.accuracy(state[0], 10)
        recall=game.recall(state[0], 30)
        map_score=game.map(state[0])
        print("Testing P@5 For "+concept+" Is "+str(accuracy))
        print("Testing P@10 For "+concept+" Is "+str(p10))
        print("Testing Recall For "+concept+" Is "+str(recall))
        print("Testing MAP For "+concept+" Is "+str(map_score))
        count+=1
        avg_accuracy+=accuracy
        avg_recall+=recall
        avg_map+=map_score
        avg_p10+=p10
    print(flag+" Average P@5 Is "+str(avg_accuracy/float(count)))
    print(flag+" Average P@10 Is "+str(avg_p10/float(count)))
    print(flag+" Average Recall Is "+str(avg_recall/float(count)))
    print(flag+" Average MAP Is "+str(avg_map/float(count)))


def word_to_vec(session, *args):
    f = open("word2vec", 'rb')
    matrix= numpy.array(pickle.load(f))
    print("word2vec shape: ", matrix.shape)
    for model in args:
        session.run(tf.assign(model.embedding, matrix))

#game environment
game = Env(use_doc=False)
#How many experiences to use for each training step.
batch_size = 100
#How often to perform a training step.
update_freq = 3
#Discount factor on the target Q-values
y = 0.001 
#Starting chance of random action
startE = 1
#Final chance of random action
endE = 0.1
#How many steps of training to reduce startE to endE.
anneling_steps = 3000
#How many episodes of game environment to train network with.
num_episodes = 60
#Rate to update target network toward primary network
tau = 0.9
#LSTM hidden size 
hidden_size = 300
#maximum steps for LSTM
num_steps=150
#maximum query size
query_max=11
#vocabulary size 
vocab_size=len(game.dictionary)+1
#Dropout rate
keep_prob=0.8
#action number
action_num=5
#training testing split 150 720
split_num=720
#whether load from an existing model
load_model=False
#saved model path
path = "./dqn"

tf.reset_default_graph()

#All history experience buffer
episodeBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:   
    #mainQN
    initializer = tf.random_normal_initializer(0, 0.05)
    with tf.variable_scope("mainQN", reuse = None, initializer=initializer):
        mainQN_train = Qnetwork(is_training=True)
    with tf.variable_scope("mainQN", reuse = True, initializer=initializer):
        mainQN_test = Qnetwork(is_training=False)
    #targetQN
    with tf.variable_scope("targetQN", reuse = None, initializer=initializer):
        targetQN_train = Qnetwork(is_training=True)
    with tf.variable_scope("targetQN", reuse = True, initializer=initializer):
        targetQN_test = Qnetwork(is_training=False)

    #Take out all trainable variables
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables,tau)

    #init all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    #initialize embedding matrix
    word_to_vec(sess, mainQN_train, targetQN_train)

    #load model
    if load_model == True:
        print('Loading Model...')
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        #load training data
        episodeBuffer.buffer= pickle.load(open('./trainingdata', 'rb'))
        
    #Set the target network to be equal to the primary network.
    updateTarget(targetOps,sess) 
    #record total steps
    total_steps=0
    for i in range(num_episodes):
        for concept in game.concepts_set[:split_num]:

            #Reset environment and get first new observation
            state = game.reset(concept)
            processed_state, state_lengths = processState(state, game)
            
            #record how long the query is for this episode (to prevent overflow length query)
            query_size=1
            while query_size<query_max:
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e:
                    print("RANDOM ACTION")
                    action = np.random.randint(0, action_num)
                else:
                    print("RATIONAL ACTION")
                    action= sess.run(mainQN_test.predict,feed_dict={mainQN_test.text_input: [processed_state], mainQN_test.text_mask:[state_lengths]})[0]
                print("ACTION: "+str(action))
                if int(action)<3:
                    query_size+=1
                state1,reward,stop = game.step(int(action))

                processed_state1, state1_lengths = processState(state1, game)
                #save the experience to our episode buffer.
                episodeBuffer.add([[processed_state,action,reward,processed_state1,stop, state_lengths, state1_lengths]]) 

                if e > endE:
                    e -= stepDrop
                total_steps+=1                
                if total_steps % (update_freq) == 0 and total_steps>batch_size+2:
                    print("-------------TRAINING--------------")
                    #Get a random batch of experiences.
                    trainBatch = episodeBuffer.sample(batch_size)
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN_test.predict,feed_dict={mainQN_test.text_input:np.vstack(trainBatch[:,3]), mainQN_test.text_mask:np.vstack(trainBatch[:,6])})
                    Q2 = sess.run(targetQN_test.Qout,feed_dict={targetQN_test.text_input:np.vstack(trainBatch[:,3]), targetQN_test.text_mask:np.vstack(trainBatch[:,6])})
                    stop_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    
                    targetQ = trainBatch[:,2] + (y*doubleQ * stop_multiplier)
                    #Update the network with our target values.
                    _, grads=sess.run([mainQN_train.updateModel, mainQN_train.grads], \
                        feed_dict={mainQN_train.text_input:np.vstack(trainBatch[:,0]), mainQN_train.text_mask:np.vstack(trainBatch[:,5]),mainQN_train.targetQ:targetQ, mainQN_train.actions:trainBatch[:,1]})                    
                    #print(grads)
                    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
                state = state1
                processed_state=processed_state1
                state_lengths=state1_lengths      
                if stop == True:
                    break
        if i%2==0:
            test_model(mainQN_test, game, game.concepts_set[:split_num], "Training")  
            test_model(mainQN_test, game, game.concepts_set[split_num:], "Testing")
            
            #save parameters
            print("SAVING MODEL")
            saver = tf.train.Saver()
            saver.save(sess,path+'/model.ckpt')
            #save training data
            pickle.dump(episodeBuffer.buffer, open('./trainingdata', 'wb'))