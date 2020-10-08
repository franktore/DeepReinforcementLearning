#### SELF PLAY
EPISODES = 90
MCTS_SIMS = 12
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 20 # turn on which it starts playing deterministically
CPUCT = 3 #1
EPSILON = 0.2
ALPHA = 1 #0.8


#### RETRAINING
BATCH_SIZE = 1024
EPOCHS = 2
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 30

HIDDEN_CNN_LAYERS = [
	{'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	 , {'filters':256, 'kernel_size': (3,3)}
	]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3
