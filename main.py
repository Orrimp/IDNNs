"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet
from idnns.networks import network_paramters as netp


def main(args):
    #Bulid the netowrk
    print ('Building the network')
    net = inet.informationNetwork(args=args)
    net.print_information()
    print ('Start running the network')
    net.run_network()
    print ('Saving data')
    net.save_data()
    print ('Ploting figures')
    #Plot the newtork
    net.plot_network()


if __name__ == '__main__':
    args = netp.get_default_parser(None)
    args.data_name = 'MNIST'
    args.learning_rate = 0.01
    args.num_ephocs = 1000
    args.interval_information_display=2
    args.num_of_samples = 50
    main(args)

