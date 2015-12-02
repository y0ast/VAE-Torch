require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
     -- The Encoder
    local encoder = nn.Sequential()
    encoder:add(nn.Linear(input_size, hidden_layer_size))
    encoder:add(nn.ReLU(true))
    
    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
    mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))

    encoder:add(mean_logvar)
    
    return encoder
end

function VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)
    -- The Decoder
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, hidden_layer_size))
    decoder:add(nn.ReLU(true))

    if continuous then
        mean_logvar = nn.ConcatTable()
        mean_logvar:add(nn.Linear(hidden_layer_size, input_size))
        mean_logvar:add(nn.Linear(hidden_layer_size, input_size))
        decoder:add(mean_logvar)
    else
        decoder:add(nn.Linear(hidden_layer_size, input_size))
        decoder:add(nn.Sigmoid(true))
    end

    return decoder
end

return VAE
