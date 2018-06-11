from idnns.forgetting.dnnmodel import DnnModel
import numpy as np


class Forgetting:

    def __init__(self):

        args = create_args()
       # layer_sizes = np.array(["784", "dropout", "400", "dropout", "100", "dropout", "50", "dropout", "10"])
        layer_sizes = np.array(["784", "400", "100", "50", "10"])

        #layer_sizes = np.array(["784", "400", "100", "50", "10"])

        network = DnnModel(layers_params=layer_sizes, args=args)
        network.create_network_layers()
        #network = network.init_information_collector(target, interval, mode)
        network.train_network()
        network.evaluate_network()

        #information = network.collect_info()

        #information.print_results()
        #information.print_information()
        #information.create_figures()


def create_args():
    C = type('type_C', (object,), {})
    args = C()
    args.learning_rate = 0.01
    args.batch_size = 200
    args.interval_to_print = 100
    args.num_of_epochs = 10

    return args


