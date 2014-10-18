function load32()
    data = {}
    data.train = torch.load('datasets/train_32x32.t7', 'ascii').data
    data.test = torch.load('datasets/test_32x32.t7', 'ascii').data

    --Convert training data to floats
    data.train = data.train:double()
    data.test = data.test:double()

    --Rescale to 0..1 and invert
    data.train:div(255):resize(60000,1024)
    data.test:div(255):resize(10000,1024)

    return data
end

function load28(path)
    local f = hdf5.open(path, 'r')

    local data = {}
    data.train = f:read('x_train'):all():double()
    data.valid = f:read('x_valid'):all():double()
    data.test = f:read('x_test'):all():double()

    return data
end

function loadfreyfaces(path)
    local f = hdf5.open(path, 'r')
    local data = {}
    data.train = f:read('train'):all():double()
    data.test = f:read('test'):all():double()

    return data
end
