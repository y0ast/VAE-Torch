require 'hdf5'

function load32()
    train = torch.load('mnist/train_32x32.t7', 'ascii')
    test = torch.load('mnist/test_32x32.t7', 'ascii')

    --Convert training data to floats
    train.data = train.data:double()
    test.data = test.data:double()

    --Rescale to 0..1 and invert
    train.data:div(255):resize(60000,1024)
    test.data:div(255):resize(10000,1024)
end

function load28(path)
    local f = hdf5.open(path, 'r')

    local data = {}
    data.train = f:read('x_train'):all():double()
    data.valid = f:read('x_valid'):all():double()
    data.test = f:read('x_test'):all():double()

    return data
end