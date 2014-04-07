-- Based on JoinTable module

require 'nn'

local Reparametrize, parent = torch.class('nn.Reparametrize', 'nn.Module')

function Reparametrize:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
end 

function Reparametrize:updateOutput(input)
    self.mu = input[1]:clone()
    self.sigma = input[2]:clone()

    -- Maybe make this batchsize dependent?
    self.eps = torch.randn(self.dimension,1)
    input[2]:mul(0.5):exp()
    
    -- Broadcast epsilon over minibatch
    input[2]:cmul(torch.expandAs(self.eps,input[2]:t()))

    self.output = torch.add(input[1],input[2])

    return self.output
end

function Reparametrize:updateGradInput(input, gradOutput)
    for i=1,#input do 
        if self.gradInput[i] == nil then
            self.gradInput[i] = input[i].new()
        end
        self.gradInput[i]:resizeAs(input[i])
    end

    -- Derivative with respect to mean is 1
    self.gradInput[1]:copy(gradOutput)
    
    -- Broadcast epsilon over gradient
    self.gradInput[2]:copy(input[2]:mul(0.5):exp():cmul(torch.expandAs(self.eps,input[2]:t())))
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
