-- Joost van Amersfoort - <joost@joo.st>
require 'sys'
require 'torch'
require 'nn'
require 'xlua'
require 'optim'

--Packages necessary for SGVB
require 'Reparametrize'
require 'GaussianCriterion'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'

--For loading data files
require 'load'


data = loadfreyfaces('datasets/freyfaces.hdf5')

dim_input = data.train:size(2)
dim_hidden = 2
hidden_units_encoder = 200
hidden_units_decoder = 200

batchSize = 20

torch.manualSeed(1)

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

--Optimization criteria
Gaussian = nn.GaussianCriterion()
KLD = nn.KLDCriterion()

parameters, gradients = va:getParameters()

config = {
    learningRate = -0.01,
}

state = {}

epoch = 0
while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train:size(1))

     --Make sure batches are always batchSize
    local N = data.train:size(1) - (data.train:size(1) % batchSize)
    local N_test = data.test:size(1) - (data.test:size(1) % batchSize)

    for i = 1, N, batchSize do
        xlua.progress(i+batchSize-1, data.train:size(1))

        local batch = torch.Tensor(batchSize,data.train:size(2))

        local k = 1
        for j = i,i+batchSize-1 do
            batch[k] = data.train[shuffle[j]]:clone() 
            k = k + 1
        end

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            va:zeroGradParameters()

            local f = va:forward(batch)
            local err = Gaussian:forward(f, batch)
            local df_dw = Gaussian:backward(f, batch)
            va:backward(batch,df_dw)

            local KLDerr = KLD:forward(va:get(1).output, batch)
            local de_dw = KLD:backward(va:get(1).output, batch)
            encoder:backward(batch,de_dw)

            local lowerbound = err  + KLDerr

            return lowerbound, gradients
        end

        x, batchlowerbound = optim.adagrad(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("\nEpoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. sys.clock() - time)

    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end

    if epoch % 2 == 0 then
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/lowerbound.t7', torch.Tensor(lowerboundlist))
    end
end
