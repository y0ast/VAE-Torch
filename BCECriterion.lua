local BCECriterion, parent = torch.class('nn.BCECriterion', 'nn.Criterion')

function BCECriterion:updateOutput(input, target)
    -- log(input) * target + log(1 - input) * (1 - target)
    self.output = torch.log(input):cmul(target)
    
    --Add the second part of error
    self.output:add(torch.add(-input,1):log():cmul(torch.add(-target,1)))

    return self.output:sum()
end

function BCECriterion:updateGradInput(input, target)
    -- target / input - (1 - target) / (1 - input)
    self.gradInput = torch.cdiv(target,input)
    self.gradInput:add(-1,torch.cdiv(torch.add(-target,1),torch.add(-input,1)))

    return self.gradInput
end
