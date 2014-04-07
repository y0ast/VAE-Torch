-- Joost van Amersfoort - <joost@joo.st>

require 'torch'
require 'nn'

require 'Reparametrize'
require 'BCECriterion'

train = torch.load('mnist.t7/train_32x32.t7', 'ascii')
test = torch.load('mnist.t7/test_32x32.t7', 'ascii')

--Convert training data to floats
train.data = train.data:double()
test.data = test.data:double()

--Rescale to 0..1 and invert
train.data:div(-255):add(1):resize(60000,1024,1)
test.data:div(-255):add(1):resize(10000,1024,1)


dim_input = 1024 --32x32
dim_hidden = 20
hidden_units_encoder = 400
hidden_units_decoder = 400

batchSize = 100

-- Check reset function of linear, uses uniform distribution
va = nn.Sequential()
va:add(nn.Linear(dim_input,hidden_units_encoder))
va:add(nn.Tanh())

c = nn.ConcatTable()
c:add(nn.Linear(hidden_units_encoder, dim_hidden))
c:add(nn.Linear(hidden_units_encoder, dim_hidden))
va:add(c)

va:add(nn.Reparametrize(dim_hidden))

--Decoding layer
va:add(nn.Linear(dim_hidden, hidden_units_decoder))
va:add(nn.Tanh())
va:add(nn.Linear(hidden_units_decoder, dim_input))
va:add(nn.Sigmoid())

--Binary cross entropy term
criterion = nn.BCECriterion()

paremeters, gradParameters = va:getParameters()

output = va:forward(train.data[1]:t())
err = criterion:forward(output, train.data[1]:t())
df_do = criterion:backward(output, train.data[1]:t())
va:backward(train.data[1]:t(),df_do)
print(gradParameters)

function train(dataset)
    prior =  torch.sum(torch.add(va:get(4).sigma,1):add(-(va:get(4).mu:pow(2))):add(-va:get(4).sigma:exp()))
    lowerbound = err + 0.5 * prior

end
