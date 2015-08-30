require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:updateOutput(input, target)
    -- -0.5 * (log(sigma) + log(2pi)) - 0.5 * (x - mu)^2/sigma^2
    -- input[1] = mu
    -- input[2] = log(sigma^2)

    local Gelement = torch.mul(input[2],0.5):add(0.5 * math.log(2 * math.pi)):mul(-1)

    Gelement:add(-1,torch.add(target,-1,input[1]):pow(2):cdiv(torch.exp(input[2])):mul(0.5))

    self.output = torch.sum(Gelement)

    return self.output
end

function GaussianCriterion:updateGradInput(input, target)
    -- Verify again for correct handling of 0.5 multiplication
	self.gradInput = {}

    -- (x - mu) / sigma^2  --> (1 / sigma^2 = exp(-log(sigma^2)) )
    self.gradInput[1] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]))

    -- - 0.25 + 0.5 * (x - mu)^2 / sigma^2
    self.gradInput[2] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]):pow(2)):add(-0.25)

    return self.gradInput
end
