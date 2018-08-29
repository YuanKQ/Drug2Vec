import sys


sys.path.extend(['/home/cdy/ykq/vae', '/home/cdy/ykq/vae/M2'])
from M2.ddi_genclass import DdiGenerativeClassifier
from M2.genclass import GenerativeClassifier
from M2.vae import VariationalAutoencoder
import numpy as np
from ddi.data_process import load_dataset_split
import data.mnist as mnist #https://github.com/dpkingma/nips14-ssl



def encode_dataset( model_path, facter_dim,hidden_layers_px, hidden_layers_qz, min_std = 0.0, ):

    # VAE = VariationalAutoencoder( dim_x = 28*28, dim_z = 50 ) #Should be consistent with model being loaded
    VAE = VariationalAutoencoder( dim_x = facter_dim, dim_z = 50,
                                  hidden_layers_px = hidden_layers_px,
                                  hidden_layers_qz = hidden_layers_qz) #Should be consistent with model being loaded
    with VAE.session:
        VAE.saver.restore( VAE.session, VAE_model_path )

        enc_x_lab_mean, enc_x_lab_var = VAE.encode( x_lab )
        enc_x_ulab_mean, enc_x_ulab_var = VAE.encode( x_ulab )
        enc_x_valid_mean, enc_x_valid_var = VAE.encode( x_valid )
        enc_x_test_mean, enc_x_test_var = VAE.encode( x_test )

        id_x_keep = np.std( enc_x_ulab_mean, axis = 0 ) > min_std

        enc_x_lab_mean, enc_x_lab_var = enc_x_lab_mean[ :, id_x_keep ], enc_x_lab_var[ :, id_x_keep ]
        enc_x_ulab_mean, enc_x_ulab_var = enc_x_ulab_mean[ :, id_x_keep ], enc_x_ulab_var[ :, id_x_keep ]
        enc_x_valid_mean, enc_x_valid_var = enc_x_valid_mean[ :, id_x_keep ], enc_x_valid_var[ :, id_x_keep ]
        enc_x_test_mean, enc_x_test_var = enc_x_test_mean[ :, id_x_keep ], enc_x_test_var[ :, id_x_keep ]

        data_lab = np.hstack( [ enc_x_lab_mean, enc_x_lab_var ] )
        data_ulab = np.hstack( [ enc_x_ulab_mean, enc_x_ulab_var ] )
        data_valid = np.hstack( [enc_x_valid_mean, enc_x_valid_var] )
        data_test = np.hstack( [enc_x_test_mean, enc_x_test_var] )

    return data_lab, data_ulab, data_valid, data_test

if __name__ == '__main__':

    #############################
    ''' Experiment Parameters '''
    #############################
    # num_lab = 10000 #int(sys.argv[1])
    if len(sys.argv) > 1:
        print(sys.argv[1])
        num_lab = int("Number of total labelled examples: ", sys.argv[1])
    else:
        num_lab = 12000         #Number of labelled examples (total)
    num_batches = 100       #Number of minibatches in a single epoch
    dim_z = 50              #Dimensionality of latent variable (z)
    epochs = 1001           #Number of epochs through the full dataset
    learning_rate = 3e-4    #Learning rate of ADAM
    alpha = 0.1             #Discriminatory factor (see equation (9) of http://arxiv.org/pdf/1406.5298v2.pdf)
    seed = 31415            #Seed for RNG

    #Neural Networks parameterising p(x|z,y), q(z|x,y) and q(y|x)
    encode_hidden_layers_px = [600, 600]
    encode_hidden_layers_qz = [600, 600]
    hidden_layers_px = [ 800, 500 ]
    hidden_layers_qz = [ 500, 500 ]
    hidden_layers_qy = [ 100, 500 ]
    print("***hidden_layers_px: ", hidden_layers_px)
    print("***hidden_layers_qz: ", hidden_layers_qz)
    print("***hidden_layers_qy: ", hidden_layers_qy)

    ####################
    ''' Load Dataset '''
    ####################

    # mnist_path = 'mnist/mnist_28.pkl.gz'
    # #Uses anglpy module from original paper (linked at top) to split the dataset for semi-supervised training
    # train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(mnist_path, binarize_y=True)
    # x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, num_lab) # num_lab的数值来调控label与unlabel的比例
    #
    # x_lab, y_lab = x_l.T, y_l.T
    # # tmp0 = x_lab[0].reshape(28, 28)
    # # tmp1 = x_lab[10].reshape(28, 28)
    # # tmp2 = x_lab[20].reshape(28, 28)
    # x_ulab, y_ulab = x_u.T, y_u.T
    # x_valid, y_valid = valid_x.T, valid_y.T
    # x_test, y_test = test_x.T, test_y.T
    test_dataset_path = "/home/cdy/ykq/vae/ddi/test_dataset"
    x_lab, y_lab, x_ulab, y_ulab, x_valid, y_valid, x_test, y_test = load_dataset_split(test_dataset_path, int(num_lab/2), 3)
    print("test_dataset_path: ", test_dataset_path)
    print("num_lab: ", num_lab)

    ################
    ''' Load VAE '''
    ################

    # VAE_model_path = '/home/yuan/Code/PycharmProjects/vae/M2/models/VAE_600-600-0.0003-50.cpkt'
    # VAE_model_path = '/home/yuan/Code/PycharmProjects/vae/checkpoints/model_VAE_0.0003-26_1526306640.2848828.cpkt'
    # VAE_model_path = '/home/yuan/Code/PycharmProjects/vae/M2/checkpoints/model_VAE_0.0003-50_1526353473.6963947.cpkt'
    VAE_model_path = '/home/cdy/ykq/vae/M2/checkpoints/model_VAE_0.0003-260_1526433033.8844438.cpkt'
    min_std = 0.1 #Dimensions with std < min_std are removed before training with GC

    data_lab, data_ulab, data_valid, data_test = encode_dataset( VAE_model_path, x_lab.shape[1], encode_hidden_layers_px, encode_hidden_layers_qz, min_std )

    dim_x = data_lab.shape[1] / 2
    dim_y = y_lab.shape[1]
    num_examples = data_lab.shape[0] + data_ulab.shape[0]
    print("dim_x", dim_x)
    print("dim_y", dim_y)

    ###################################
    ''' Train Generative Classifier '''
    ###################################

    GC = GenerativeClassifier(  dim_x, dim_z, dim_y,
                                num_examples, num_lab, num_batches,
                                hidden_layers_px    = hidden_layers_px,
                                hidden_layers_qz    = hidden_layers_qz,
                                hidden_layers_qy    = hidden_layers_qy,
                                alpha               = alpha )

    GC.train(   x_labelled      = data_lab, y = y_lab, x_unlabelled = data_ulab,
                x_valid         = data_valid, y_valid = y_valid,
                epochs          = epochs,
                learning_rate   = learning_rate,
                seed            = seed,
                print_every     = 10,
                load_path       = None )


    ############################
    ''' Evaluate on Test Set '''
    ############################

    GC_eval = GenerativeClassifier(  dim_x, dim_z, dim_y, num_examples, num_lab, num_batches,
                                     hidden_layers_px=hidden_layers_px,
                                     hidden_layers_qz=hidden_layers_qz,
                                     hidden_layers_qy=hidden_layers_qy,
                                     )

    with GC_eval.session:
        GC_eval.saver.restore( GC_eval.session, GC.save_path )
        GC_eval.predict_labels( data_test, y_test )
