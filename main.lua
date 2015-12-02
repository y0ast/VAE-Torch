-- Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

nngraph.setDebug(false)

local VAE = require 'VAE'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'

--For loading data files
require 'load'

local continuous = false
data = load(continuous)

local input_size = data.train:size(2)
local latent_variable_size = 20
local hidden_layer_size = 400

local batch_size = 100

torch.manualSeed(1)

local encoder = VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
local decoder = VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local reconstruction, reconstruction_var, model
if continuous then
    reconstruction, reconstruction_var = decoder(z):split(2)
    model = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})
    criterion = nn.GaussianCriterion()
else
    reconstruction = decoder(z)
    model = nn.gModule({input},{reconstruction, mean, log_var})
    criterion = nn.BCECriterion()
    criterion.sizeAverage = false
end

-- Some code to draw computational graph
-- dummy_x = torch.rand(dim_input)
-- model:forward({dummy_x})

-- Uncomment to get structure of the Variational Autoencoder
-- graph.dot(.fg, 'Variational Autoencoder', 'VA')

KLD = nn.KLDCriterion()

local parameters, gradients = model:getParameters()

local config = {
    learningRate = 0.001
}

local state = {}

epoch = 0
while true do
    epoch = epoch + 1
    local lowerbound = 0
    local tic = torch.tic()

    local shuffle = torch.randperm(data.train:size(1))

    -- This batch creation is inspired by szagoruyko CIFAR example.
    local indices = torch.randperm(data.train:size(1)):long():split(batch_size)
    indices[#indices] = nil
    local N = #indices * batch_size

    local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = data.train:index(1,v)

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            model:zeroGradParameters()
            local reconstruction, reconstruction_var, mean, log_var
            if continuous then
                reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(inputs))
                reconstruction = {reconstruction, reconstruction_var}
            else
                reconstruction, mean, log_var = unpack(model:forward(inputs))
            end

            local err = criterion:forward(reconstruction, inputs)
            local df_dw = criterion:backward(reconstruction, inputs)

            local KLDerr = KLD:forward(mean, log_var)
            local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))

            if continuous then
                error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}
            else
                error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
            end

            model:backward(inputs, error_grads)

            local batchlowerbound = err + KLDerr

            return batchlowerbound, gradients
        end

        x, batchlowerbound = optim.adam(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. torch.toc(tic)) 

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
