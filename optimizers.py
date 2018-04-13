import torch.optim as optim

ADADELTA_LEARNING_RATE = 0.05
ADADELTA_MOMENTUM = 0.9
ADADELTA_WEIGHT_DECAY = 0.005

def get_adadelta_halnet(halnet,
                        momentum=ADADELTA_MOMENTUM,
                        weight_decay=ADADELTA_WEIGHT_DECAY,
                        learning_rate=ADADELTA_LEARNING_RATE):
    return optim.Adadelta(halnet.parameters(),
                               rho=momentum,
                               weight_decay=weight_decay,
                               lr=learning_rate)
