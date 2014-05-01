-- Joost van Amersfoort - <joost@joo.st>
require 'sys'
require 'torch'
require 'nn'
require 'xlua'

--Packages necessary for SGVB
require 'Reparametrize'
require 'BCECriterion'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'

--For loading data files
require 'load'

--For saving weights and biases
require 'hdf5'

require 'adagrad'

data = load28('datasets/mnist.hdf5')

dim_input = data.train:size(2) 
dim_hidden = 20
hidden_units_encoder = 400
hidden_units_decoder = 400

batchSize = 100
learningRate = 0.03

adaGradInitRounds = 10

torch.manualSeed(1)
--Does not seem to do anything
torch.setnumthreads(2)

--The model

--Encoding layer
encoder = nn.Sequential()
encoder:add(nn.LinearVA(dim_input,hidden_units_encoder))
encoder:add(nn.Tanh())

z = nn.ConcatTable()
z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))
z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))

encoder:add(z)

va = nn.Sequential()
va:add(encoder)

--Reparametrization step
va:add(nn.Reparametrize(dim_hidden))

--Decoding layer
va:add(nn.LinearVA(dim_hidden, hidden_units_decoder))
va:add(nn.Tanh())
va:add(nn.LinearVA(hidden_units_decoder, dim_input))
va:add(nn.Sigmoid())

--Binary cross entropy term
BCE = nn.BCECriterion()
KLD = nn.KLDCriterion()

opfunc = function(batch) 
    va:zeroGradParameters()

    f = va:forward(batch)
    err = BCE:forward(f, batch)
    df_dw = BCE:backward(f, batch)
    va:backward(batch,df_dw)

    KLDerr = KLD:forward(va:get(1).output, batch)
    de_dw = KLD:backward(va:get(1).output, batch)
    encoder:backward(batch,de_dw)


    lowerbound = err  + KLDerr
    weights, grads = va:parameters()

    return weights, grads, lowerbound
end

h = adaGradInit(data.train, opfunc, adaGradInitRounds)


epoch = 0
while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train:size(1))

    for i = 1, data.train:size(1), batchSize do
        local iend = math.min(data.train:size(1),i+batchSize-1)
        xlua.progress(iend, data.train:size(1))

        local batch = torch.Tensor(iend-i+1,data.train:size(2))

        local k = 1
        for j = i,iend do
            batch[k] = data.train[shuffle[j]]:clone() 
            k = k + 1
        end

        batchlowerbound = adaGradUpdate(batch, opfunc)
        lowerbound = lowerbound + batchlowerbound
    end

    print("\nEpoch: " .. epoch .. " Lowerbound: " .. lowerbound/data.train:size(1) .. " time: " .. sys.clock() - time)
    if epoch % 2 == 0 then
        local myFile = hdf5.open('params/epoch_' .. epoch .. '.hdf5', 'w')

        myFile:write('weighttanh', va:get(3).weight)
        myFile:write('biastanh', va:get(3).bias)
        myFile:write('weightsigmoid', va:get(5).weight)
        myFile:write('biassigmoid', va:get(5).bias)

        myFile:close()
    end

end
