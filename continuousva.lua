-- Joost van Amersfoort - <joost@joo.st>
require 'sys'
require 'torch'
require 'nn'
require 'xlua'

--Packages necessary for SGVB
require 'Reparametrize'
require 'GaussianCriterion'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'

--For loading data files
require 'load'

require 'adagrad'

torch.manualSeed(1)
--Does not seem to do anything
torch.setnumthreads(2)


data = loadfreyfaces('datasets/freyfaces.hdf5')
dim_input = data.train:size(2)

dim_hidden = 5
hidden_units_encoder = 200
hidden_units_decoder = 200

batchSize = 20
learningRate = 0.03

adaGradInitRounds = 10


--The model

--Encoding layer
encoder = nn.Sequential()
encoder:add(nn.LinearVA(dim_input,hidden_units_encoder))

encoder:add(nn.SoftPlus())

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
va:add(nn.SoftPlus())

decoder = nn.ConcatTable()
decoder:add(nn.LinearVA(hidden_units_decoder, dim_input))
decoder:add(nn.LinearVA(hidden_units_decoder, dim_input))

decoder2 = nn.ParallelTable()
decoder2:add(nn.Sigmoid())
decoder2:add(nn.Copy())

va:add(decoder)
va:add(decoder2)

--Binary cross entropy term
Gaussian = nn.GaussianCriterion()
KLD = nn.KLDCriterion()

opfunc = function(batch) 
    va:zeroGradParameters()

    f = va:forward(batch)
    err = Gaussian:forward(f, batch)
    df_dw = Gaussian:backward(f, batch)
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
    for i = 1, data.train:size(1), batchSize do
        xlua.progress(i+batchSize-1, data.train:size(1))
        batch = data.train[{{i,math.min(data.train:size(1),i+batchSize-1)}}]

        batchlowerbound = adaGradUpdate(batch, opfunc)
        lowerbound = lowerbound + batchlowerbound
    end

    print("\nEpoch: " .. epoch .. " Lowerbound: " .. lowerbound/data.train:size(1) .. " time: " .. sys.clock() - time)
end
