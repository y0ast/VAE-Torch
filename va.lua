-- Joost van Amersfoort - <joost@joo.st>

require 'torch'
require 'nn'
require 'hdf5'

require 'Reparametrize'
require 'BCECriterion'

--Custom Linear to support different reset function
require 'LinearVA'

function loadTorch()
    train = torch.load('mnist/train_32x32.t7', 'ascii')
    test = torch.load('mnist/test_32x32.t7', 'ascii')

    --Convert training data to floats
    train.data = train.data:double()
    test.data = test.data:double()

    --Rescale to 0..1 and invert
    train.data:div(255):resize(60000,1024)
    test.data:div(255):resize(10000,1024)
end

function loadTheano()
    local f = hdf5.open('mnist/mnist.hdf5', 'r')

    train = {}
    train.data = f:read('x_train'):all():double()
    print(train)

    valid = {}
    valid.data = f:read('x_valid'):all():double()

    test = {}
    test.data = f:read('x_test'):all():double()
end

loadTheano()

dim_input = train.data:size(2) 
dim_hidden = 20
hidden_units_encoder = 200
hidden_units_decoder = 200

batchSize = 100

learningRate = 0.03

va = nn.Sequential()
va:add(nn.LinearVA(dim_input,hidden_units_encoder))
va:add(nn.Tanh())

c = nn.ConcatTable()
c:add(nn.LinearVA(hidden_units_encoder, dim_hidden))
c:add(nn.LinearVA(hidden_units_encoder, dim_hidden))
va:add(c)

va:add(nn.Reparametrize(dim_hidden))

--Decoding layer
va:add(nn.LinearVA(dim_hidden, hidden_units_decoder))
va:add(nn.Tanh())
va:add(nn.LinearVA(hidden_units_decoder, dim_input))
va:add(nn.Sigmoid())

--Binary cross entropy term
criterion = nn.BCECriterion()

h = {}

for i = 1,5000, batchSize do
    batch = train.data[{{i,i+batchSize-1}}]

    va:zeroGradParameters()

    output = va:forward(batch)
    err = criterion:forward(output, batch)
    df_dw = criterion:backward(output, batch)
    va:backward(batch,df_dw)

    weights, grads = va:parameters()

    for i=1,#grads do
        if h[i] == nil then
            h[i] = torch.cmul(grads[i],grads[i]):add(0.01)
        else
            h[i]:add(torch.cmul(grads[i],grads[i]))
        end
    end
end


print("AdaGrad matrix initialized")

function printWeights()
    print("Weights")
    weights, grads = va:parameters()
    for j=1,#weights do
        print(torch.norm(weights[j]))
    end
    print("grads")
    for j=1,#grads do
        print(torch.norm(grads[j]))
    end
end
function updateParameters(AdaGrad)
    if AdaGrad then
        weights, grads = va:parameters()
        for i=1,#h do
            h[i]:add(torch.cmul(grads[i],grads[i]))
            if i % 2 == 0 then
                prior = 0
            else
                prior = -torch.mul(weights[i],0.5):mul(batchSize/50000)
            end

            update = torch.Tensor(h[i]:size()):fill(learningRate)
            update:cdiv(h[i]):cmul(torch.add(grads[i],prior))

            weights[i]:add(update)
        end
    else
        va:updateParameters(-learningRate/batchSize)
    end
end

function run(dataset)
    local lowerbound = 0
    for i = 1, dataset:size(1), batchSize do
        batch = dataset[{{i,i+batchSize-1}}]

        va:zeroGradParameters()

        output = va:forward(batch)
        err = criterion:forward(output, batch)
        df_dw = criterion:backward(output, batch)
        va:backward(batch,df_dw)

        -- sum(1 + log(sigma^2) - mu^2 - sigma^2)
        prior =  torch.sum(torch.add(va:get(4).sigma,1):add(-1,torch.pow(va:get(4).mu,2)):add(-torch.exp(va:get(4).sigma)))

        batchlowerbound =  err + 0.5 * prior
        lowerbound = lowerbound + batchlowerbound

        updateParameters(False)
        -- printWeights()
        -- io.read()

        print(i, batchlowerbound/batchSize)
        if batchlowerbound/batchSize < -1000 then 
            printWeights()
            os.exit()
        end

        collectgarbage()
    end
    print("lowerbound", lowerbound/50000)
end

while true do
    run(train.data)
end
