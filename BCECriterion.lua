local BCECriterion, parent = torch.class('nn.BCECriterion', 'nn.Criterion')

function BCECriterion:updateOutput(input, target)
    self.output = (input * target:log()) + (1 - input) * (1 - target:log())

    return self.output
end

function BCECriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)

    self.gradInput = target / input - (1 - target) / (1 - input)

    return self.gradInput
end
