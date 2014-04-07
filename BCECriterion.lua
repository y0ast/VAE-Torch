local BCECriterion, parent = torch.class('nn.BCECriterion', 'nn.Criterion')

function BCECriterion:updateOutput(input, target)
    self.output = torch.cmul(input:log(),target)
    
    self.output:add(torch.add(-input,1):cmul(torch.add(-target,1)))

    return self.output:sum()
end

function BCECriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)

    -- x / y - (1 - x) / (1 - y)
    self.gradInput = torch.cdiv(target,input)
    self.gradInput:add(-(torch.add(-target,1):cdiv(torch.add(-input,1))))

    return self.gradInput
end
