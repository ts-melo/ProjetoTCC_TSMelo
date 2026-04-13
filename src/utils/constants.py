
PATH            = 'C:/Users/thais/OneDrive/Documents/Thesis/'                          
DATA_FOLDER     = PATH + 'data/'
OUTPUT_FOLDER   = PATH + 'output/'
LOG_FOLDER      = PATH + 'log/'

DATASET_FILE    = DATA_FOLDER + 'merged_dataset.csv'
ONLINE_DATASET_FILE = DATA_FOLDER + 'Tuesday-WorkingHours.pcap_ISCX.csv'

LABEL_COLUMN    = ' Label'          

BENIGN_LABEL    = 'BENIGN'

TEST_SIZE       = 0.2               
RANDOM_STATE    = 42

MODELS          = ['decision_tree', 'random_forest', 'mlp']

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = None              

DT_MAX_DEPTH    = None

CLASSIFICATION_MODE = 'both'

POISSON_RATE    = 10.0              
SIM_STEPS       = 100               

MLP_HIDDEN_LAYERS = (100, 50)
MLP_MAX_ITER = 300
MLP_ACTIVATION = 'relu'

