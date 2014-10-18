local BCECriterion2, parent = torch.class('nn.BCECriterion2', 'nn.Criterion')

function BCECriterion2:updateOutput(input, target)
    -- log(input) * target + log(1 - input) * (1 - target)
    self.output = torch.log(input):cmul(target)
    
    self.output:add(torch.add(-input,1):log():cmul(torch.add(-target,1)))

    return self.output:sum()
end

function BCECriterion2:updateGradInput(input, target)
    -- target / input - (1 - target) / (1 - input)
    self.gradInput = torch.cdiv(target,input)
    self.gradInput:add(-1,torch.cdiv(torch.add(-target,1),torch.add(-input,1)))

    return self.gradInput
end
