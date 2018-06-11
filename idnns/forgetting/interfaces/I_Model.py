

class I_Model(object):

    def create_network_layers(self):
        raise NotImplementedError("Extend I_Model and implement method create_network_layers")

    def fill_feed_dict(self, traing_data=True):
        raise NotImplementedError("Extend I_Model and implement method fill_feed_dict")

    def train_network(self):
        raise NotImplementedError("Extend I_Model and implement method train_network")

    def evaluate_network(self, num_of_epochs):
        raise NotImplementedError("Extend I_Model and implement method evaluate_network")
