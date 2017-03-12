class RNNModel():
    def __init__(self, trainX, trainY, devX, devY, config):
        trainXShape = trainX.shape
        devXShape = trainY.shape

        # Make sure data matches
        assert(trainXShape[1] == devXShape[1], )

        self.nTrain = trainX.shape[0]
        self.nFeats = trainX.shape[1]