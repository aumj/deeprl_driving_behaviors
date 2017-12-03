import numpy as np
from tqdm import tqdm
import tensorflow as tf
import imageio
imageio.plugins.ffmpeg.download()
from gym_traffic.utils.helper import *
from gym_traffic.agents.drqn import DRQN
from IPython import embed


class DRQNTester(object):

  def __init__(self):
    # self.max_steps_per_episode=max_steps_per_episode
          #Setting the training parameters
    # self.batch_size = 4 #How many experience traces to use for each training step.
    # self.trace_length = 8 #How long each experience trace will be when training
    # self.update_freq = 5 #How often to perform a training step.
    # self.y = .99 #Discount factor on the target Q-values
    # self.startE = 1 #Starting chance of random action
    # self.endE = 0.1 #Final chance of random action
    # self.anneling_steps = 10000 #How many steps of training to reduce startE to endE.
    # self.num_episodes = 10000 #How many episodes of game environment to train network with.
    # self.pre_train_steps = 10000 #How many steps of random actions before training begins.
    # self.load_model = False #Whether to load a saved model.
    # self.path = "./drqn" #The path to save our model to.
    # self.h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    # self.max_epLength = max_steps_per_episode #The max allowed length of our episode.
    # self.time_per_step = 1 #Length of each step used in gif creation
    # self.summaryLength = 100 #Number of epidoes to periodically save for analysis
    # self.tau = 0.001

    self.e = 0.01 #The chance of chosing a random action
    self.num_episodes = 10000 #How many episodes of game environment to train network with.
    self.load_model = True #Whether to load a saved model.
    self.path = "./drqn" #The path to save/load our model to/from.
    self.h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    self.max_epLength = 50 #The max allowed length of our episode.
    self.time_per_step = 1 #Length of each step used in gif creation
    self.summaryLength = 100 #Number of epidoes to periodically save for analysis


  def run_testing(self, env, nb_epoch, nb_episodes = 100, render=True, verbose=True, train=True):
    tf.reset_default_graph()
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    mainQN = Qnetwork(h_size,cell,'main')
    targetQN = Qnetwork(h_size,cellT,'target')

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=2)

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    #Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    ##Write the first line of the master log-file for the Control Center
    with open('./Center/log.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])    
        
        #wr = csv.writer(open('./Center/log.csv', 'a'), quoting=csv.QUOTE_ALL)
    with tf.Session() as sess:
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(init)

            
        for i in range(num_episodes):
            episodeBuffer = []
            #Reset environment and get first new observation
            sP = env.reset()
            s = processState(sP)
            d = False
            rAll = 0
            j = 0
            state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
            #The Q-Network
            while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e:
                    state1 = sess.run(mainQN.rnn_state,\
                        feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,mainQN.state_in:state,mainQN.batch_size:1})
                    a = np.random.randint(0,4)
                else:
                    a, state1 = sess.run([mainQN.predict,mainQN.rnn_state],\
                        feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,\
                        mainQN.state_in:state,mainQN.batch_size:1})
                    a = a[0]
                s1P,r,d = env.step(a)
                s1 = processState(s1P)
                total_steps += 1
                episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
                rAll += r
                s = s1
                sP = s1P
                state = state1
                if d == True:

                    break

            bufferArray = np.array(episodeBuffer)
            jList.append(j)
            rList.append(rAll)

            #Periodically save the model. 
            if len(rList) % summaryLength == 0 and len(rList) != 0:
                print (total_steps,np.mean(rList[-summaryLength:]), e)
                saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
                    summaryLength,h_size,sess,mainQN,time_per_step)
    print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")