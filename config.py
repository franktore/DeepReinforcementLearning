#### SELF PLAY
EPISODES = 30
MCTS_SIMS = 30
MEMORY_SIZE = 3000
TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
CPUCT = 4
EPSILON = 0.2
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 1024
EPOCHS = 20
REG_CONST = 0.0001
LEARNING_RATE = 0.06
MOMENTUM = 0.9
TRAINING_LOOPS = 1

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
