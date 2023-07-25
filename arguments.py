import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train', action='store_true', default=True, help='train the model')
    parser.add_argument('--model', type=str, default="LeNet5", help='model name') #PDGNN
    parser.add_argument('--nr_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--device', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--add-noise', action='store_true', default=True, help='add noise to the image')
    parser.add_argument('--pad-image', action='store_true', default=True, help='pad the image')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--save-path', type=str, default='saved_models/', help='path to save the model')
    parser.add_argument('--noise-factor', type=float, default=0.01, help='factor to multiply the noise with')
    parser.add_argument('--max-padding', type=int, default=80, help='maximum padding to add to the image')
    return parser.parse_args()