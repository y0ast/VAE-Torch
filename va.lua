-- Joost van Amersfoort - <joost@joo.st>

require 'torch'
require 'nn'

require 'Reparametrize'
require 'BCECriterion'

--Custom Linear to support different reset function
require 'LinearVA'

train = torch.load('mnist.t7/train_32x32.t7', 'ascii')
test = torch.load('mnist.t7/test_32x32.t7', 'ascii')

--Convert training data to floats
train.data = train.data:double()
test.data = test.data:double()

--Rescale to 0..1 and invert
train.data:div(-255):add(1):resize(60000,1024)
test.data:div(-255):add(1):resize(10000,1024)


dim_input = 1024 --32x32
dim_hidden = 20
hidden_units_encoder = 200
hidden_units_decoder = 200

batchSize = 10

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

parameters, gradParameters = va:getParameters()

function run(dataset)
    local lowerbound = 0
    for i = 1, 50000, batchSize do
        batch = dataset[{{i,i+batchSize-1}}]

        gradParameters:zero()

        output = va:forward(batch)

        err = criterion:forward(output, batch)
        df_dw = criterion:backward(output, batch)
        va:backward(batch,df_dw)

        -- sum(1 + log(sigma^2) - mu^2 - sigma^2)
        prior =  torch.sum(torch.add(va:get(4).sigma,1):add(-(va:get(4).mu:pow(2))):add(-va:get(4).sigma:exp()))
        batchlowerbound =  err + 0.5 * prior
        lowerbound = lowerbound + batchlowerbound
        -- for j=1,8 do
        --     if type(va:get(j).weight) ~= 'nil' then
        --         print(torch.norm(va:get(j).gradWeight))
        --     end
        -- end
        -- print("-----------")
        -- for j=1,2 do
        --     if type(c:get(j).weight) ~= 'nil' then
        --         print(torch.norm(c:get(j).gradWeight))
        --     end
        -- end
        -- print("-----------")
        print(i, batchlowerbound/batchSize)

        va:updateParameters(-0.03/batchSize)

    end
    print("------------------")
    print(lowerbound/50000)
end

while true do
    run(train.data)
end
