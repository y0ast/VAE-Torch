local LinearVA, parent = torch.class('nn.LinearVA', 'nn.Linear')

--Custom reset function
function LinearVA:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
end

function LinearVA:reset()
    sigmaInit = 0.01
    self.weight:normal(0, 0.01)
    self.bias:normal(0, 0.01)
end
