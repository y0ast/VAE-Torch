function load(continuous)
    if continuous then
        return loadfreyfaces()
    else
        return loadBinarizedMNIST()
    end
end

--Read MNIST file downloaded and return Tensor  
function readMNISTfile(fname,lines)
    local data = torch.Tensor(lines,784):fill(0)
    local f    = torch.DiskFile(fname,'r')
    for i=1,lines do 
        data[i] = torch.Tensor(f:readDouble(784))
    end
    return data
end

--Download data and setup directory
function getBinarizedMNIST()
    --Get train & valid. Append them
    if not paths.dirp('./binarizedMNIST') then 
        paths.mkdir('./binarizedMNIST')
    end
    print ('Downloading data...')
    os.execute('wget -O ./binarizedMNIST/binarized_mnist_train.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat')        
    os.execute('wget -O ./binarizedMNIST/binarized_mnist_valid.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat') 
    os.execute('wget -O ./binarizedMNIST/binarized_mnist_test.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat')
    print ('Converting data to torch format...')
    test  = readMNISTfile('./binarizedMNIST/binarized_mnist_test.amat',10000)
    train = readMNISTfile('./binarizedMNIST/binarized_mnist_train.amat',50000)
    valid = readMNISTfile('./binarizedMNIST/binarized_mnist_valid.amat',10000)
    print ('Saving data...')
    torch.save('./binarizedMNIST/train.t7',train)
    torch.save('./binarizedMNIST/test.t7',test)
    torch.save('./binarizedMNIST/valid.t7',valid)
end

--Load standard MNIST data
function loadBinarizedMNIST()
    if not paths.dirp('./binarizedMNIST') or not paths.filep('./binarizedMNIST/valid.t7') or not paths.filep('./binarizedMNIST/test.t7') or not paths.filep('./binarizedMNIST/train.t7') then 
        getBinarizedMNIST()
    end
    print ('Loading Binarized MNIST dataset')
    local train = torch.load('./binarizedMNIST/train.t7')
    local test  = torch.load('./binarizedMNIST/test.t7')
    local valid = torch.load('./binarizedMNIST/valid.t7')
    local dataset = {}
    data.train = torch.cat(train,valid,1)
    data.test  = test
    dataset.dim_input = 784

    collectgarbage()
    return data
end


function loadfreyfaces()
    require 'hdf5'
    local f = hdf5.open('datasets/freyfaces.hdf5', 'r')
    local data = {}
    data.train = f:read('train'):all():double()
    data.test = f:read('test'):all():double()
    f:close()

    return data
end
