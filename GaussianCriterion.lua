require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:updateOutput(input, target)
    local Gelement = torch.mul(input[2],0.5):add(0.5 * math.log(2 * math.pi)):mul(-1)
    Gelement:add(-1,torch.add(target,-1,input[1]):cdiv(torch.mul(input[2],0.5):exp()):pow(2):mul(0.5))

    self.output = torch.sum(Gelement)
    return self.output
end

function GaussianCriterion:updateGradInput(input, target)
	-- Fix sigma for not begin multiplied by 0.5!
	self.gradInput = {}
    self.gradInput[1] = torch.mul(input[2], -2):exp():mul(2):cmul(torch.add(target,-1,input[1])):div(2)
    self.gradInput[2] = torch.mul(input[2], -2):exp():cmul(torch.add(target,-1,input[1]):pow(2)):add(-1)

    return self.gradInput
end
