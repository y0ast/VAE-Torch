require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:updateOutput(input, target)
    -- Verify again for correct handling of 0.5 multiplication
    local Gelement = torch.add(input[2],math.log(2 * math.pi)):mul(-0.5)
    Gelement:add(-1,torch.add(target,-1,input[1]):pow(2):cdiv(torch.exp(input[2])):mul(0.5))

    self.output = torch.sum(Gelement)
    return self.output
end

function GaussianCriterion:updateGradInput(input, target)
    -- Verify again for correct handling of 0.5 multiplication
	self.gradInput = {}
    self.gradInput[1] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]))
    self.gradInput[2] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]):pow(2)):add(-0.5)

    return self.gradInput
end
