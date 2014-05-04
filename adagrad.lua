function adaGradInit(data, opfunc, adaGradInitRounds)
    h = {}
    for i = 1, batchSize*adaGradInitRounds+1, batchSize do
        local batch = data[{{i,i+batchSize-1}}]

        weights, grads, lowerbound = opfunc(batch)

        for j=1,#grads do
            if h[j] == nil then
                h[j] = torch.cmul(grads[j],grads[j]):add(0.01)
            else
                h[j]:add(torch.cmul(grads[j],grads[j]))
            end
        end
    end

    collectgarbage()
    return h
end

function adaGradUpdate(batch, opfunc, h)
    weights, grads, lowerbound = opfunc(batch)

    for i=1,#h do
        h[i]:add(torch.cmul(grads[i],grads[i]))

        local prior = 0
        if i % 2 ~= 0 then
            prior = -torch.mul(weights[i],0.5):mul(batchSize/data.train:size(1))
        end

        local update = torch.Tensor(h[i]:size()):fill(learningRate)
        update:cdiv(torch.sqrt(h[i])):cmul(torch.add(grads[i],prior))

        weights[i]:add(update)
    end

    collectgarbage()
    return lowerbound
end
