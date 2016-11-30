local KLDPenalty, parent = torch.class('nn.KLDPenalty', 'nn.Module')

function KLDPenalty:updateOutput(input)
    local mean, log_var = table.unpack(input)
    self.output = input

    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()
    KLDelements:exp():mul(-1)
    KLDelements:add(-1, torch.pow(mean, 2))
    KLDelements:add(1)
    KLDelements:add(log_var)
    self.loss = -0.5 * torch.sum(KLDelements)

    return self.output
end

function KLDPenalty:updateGradInput(input, gradOutput)
    assert(#gradOutput == 2)
    local mean, log_var = table.unpack(input)
    self.gradInput = {}
    self.gradInput[1] = mean:clone() + gradOutput[1]

    -- Fix this to be nicer
    self.gradInput[2] = torch.exp(log_var):mul(-1):add(1):mul(-0.5) + gradOutput[2]

    return self.gradInput
end
