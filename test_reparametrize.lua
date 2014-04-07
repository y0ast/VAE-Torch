require 'nn'
require 'Reparametrize'

print '==> testing backprop with Jacobian (finite element)'

-- to test the module, we need to freeze the randomness,
-- as the Jacobian tester expects the output of a module
-- to be deterministic...
-- so the code is the same, except that we only generate
-- the random noise once, for the whole test.
firsttime = true

-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 20
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.Reparametrize(20)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
