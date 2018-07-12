import idnns.forgetting.information.probability
import idnns.forgetting.information.entropy


class Information:

    def __init__(self, network, store):
        """
        :param network:
        :param store:
        """

        self.network = network
        self.store = store


    def calc_information_layer(self, layer_index):
        """Compute the information of a single layer"""

        return None


    def calc_information_neurons(self, neuron_indexes):
        """
        Compute the information of the given neuronen given as indexes
        :param neuron_indexes:   array of neuron indexes without input and output
        :return: information IX and IY
        """
        IX = {}
        IY = {}

        return {'IX': IX, "IY": IY}


    def calc_informtion_all_neurons_from_output(self, neuron_indexes):
        """
        Compute the information of all neurons activated by the output neuron
        :param neuron_indexes:
        :return:
        """
        return None


    def calc_information_all_activated_neurons(self, neuron_indexes):
        """
        Compute the information of all activated neurons
        :param neuron_indexes:
        :return:
        """

        return None


    def calc_information_one_neuron(self, neuron):
        """Calculcate Inforamtion (Entropy) of a single neuron with all his weights and bias values"""

