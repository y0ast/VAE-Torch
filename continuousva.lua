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

--For saving weights and biases
require 'hdf5'

require 'adagrad'

torch.manualSeed(1)
--Does not seem to do anything
torch.setnumthreads(2)


data = loadfreyfaces('datasets/freyfaces.hdf5')
dim_input = data.train:size(2)

dim_hidden = 2
hidden_units_encoder = 200
hidden_units_decoder = 200

batchSize = 20
learningRate = 0.05

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


--Continue old training session
continue = true
epoch = 4

if continue then
    path = 'params/ff_epoch_' .. epoch .. '.hdf5'

    h = torch.load('params/adagrad_' .. epoch)

    weights, bias = va:getParameters()
    weights:copy(torch.load('params/model_' .. epoch))
else
    epoch = 0
    h = adaGradInit(data.train, opfunc, adaGradInitRounds)
end


--The main loop
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

        batchlowerbound = adaGradUpdate(batch, opfunc, h)
        lowerbound = lowerbound + batchlowerbound
    end

    print("\nEpoch: " .. epoch .. " Lowerbound: " .. lowerbound/data.train:size(1) .. " time: " .. sys.clock() - time)
    if epoch % 2 == 0 then
        local myFile = hdf5.open('params/ff_epoch_' .. epoch .. '.hdf5', 'w')

        myFile:write('wrelu', va:get(3).weight)
        myFile:write('brelu', va:get(3).bias)
        myFile:write('wsig', decoder:get(1).weight)
        myFile:write('bsig', decoder:get(1).bias)
        myFile:close()

        weights, bias = va:getParameters()
        torch.save('params/model_' .. epoch, weights)
        torch.save('params/adagrad_' .. epoch, h)

    end
end
