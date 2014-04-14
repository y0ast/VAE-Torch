-- Joost van Amersfoort - <joost@joo.st>

require 'torch'
require 'nn'
require 'hdf5'

require 'Reparametrize'
require 'BCECriterion'
require 'KLDCriterion'


--Custom Linear to support different reset function
require 'LinearVA'

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

function load28()
    local f = hdf5.open('mnist/mnist.hdf5', 'r')

    train = {}
    train.data = f:read('x_train'):all():double()

    valid = {}
    valid.data = f:read('x_valid'):all():double()

    test = {}
    test.data = f:read('x_test'):all():double()
end

load28()

dim_input = train.data:size(2) 
dim_hidden = 20
hidden_units_encoder = 400
hidden_units_decoder = 400

batchSize = 100

learningRate = 0.03

torch.manualSeed(1)

encoder = nn.Sequential()
encoder:add(nn.LinearVA(dim_input,hidden_units_encoder))
encoder:add(nn.Tanh())

z = nn.ConcatTable()
z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))
z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))

encoder:add(z)

va = nn.Sequential()
va:add(encoder)
va:add(nn.Reparametrize(dim_hidden))

--Decoding layer
va:add(nn.LinearVA(dim_hidden, hidden_units_decoder))
va:add(nn.Tanh())
va:add(nn.LinearVA(hidden_units_decoder, dim_input))
va:add(nn.Sigmoid())

--Binary cross entropy term
criterion = nn.BCECriterion()

KLD = nn.KLDCriterion()

function printWeights()
    -- print("Weights")
    -- weights, grads = va:parameters()
    -- for j=1,#weights do
        -- print(torch.norm(weights[j]))
    -- end
    print("grads")
    for j=1,#grads do
        print(torch.norm(grads[j]))
    end
end

h = {}

for i = 1,1000, batchSize do
    batch = train.data[{{i,i+batchSize-1}}]

    va:zeroGradParameters()

    output = va:forward(batch)
    err = criterion:forward(output, batch)
    df_dw = criterion:backward(output, batch)
    va:backward(batch,df_dw)

    prior = KLD:forward(va:get(1).output, batch)
    dp_dw = KLD:backward(va:get(1).output, batch)
    encoder:backward(batch,dp_dw)

    weights, grads = va:parameters()

    for i=1,#grads do
        if h[i] == nil then
            h[i] = torch.cmul(grads[i],grads[i]):add(0.01)
        else
            h[i]:add(torch.cmul(grads[i],grads[i]))
        end
    end
end

collectgarbage()

print("AdaGrad matrix initialized")


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

        prior = KLD:forward(va:get(1).output, batch)
        dp_dw = KLD:backward(va:get(1).output, batch)
        encoder:backward(batch,dp_dw)

        weights, grads = va:parameters()
        printWeights()

        batchlowerbound =  err + prior
        lowerbound = lowerbound + batchlowerbound

        updateParameters(True)
        print(i, err)
        print(i, prior)
        io.read()

        if batchlowerbound/batchSize < -1000 then 
            printWeights()
            os.exit()
        end

        collectgarbage()
    end
    print("lowerbound", lowerbound/dataset:size(1))
end

while true do
    run(train.data)
end
