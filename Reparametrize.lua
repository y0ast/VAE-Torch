-- Based on JoinTable module
local Reparametrize, parent = torch.class('nn.Reparametrize', 'nn.Module')

function Reparametrize:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
end 

function Reparametrize:updateOutput(input)
    self.mu = input[1]
    self.sigma = input[2]

    self.eps = torch.randn(self.dimension)
    input[2]:mul(0.5):exp()
    input[2]:cmul(self.eps)



    self.output = input[1]:add(input[2])

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
    
    --Check if sigma is correct here
    self.gradInput[2]:copy(input[2]:mul(0.5):exp()):cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
