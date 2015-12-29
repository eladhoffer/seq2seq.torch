require 'optim'
require 'eladtools'
require 'nn'
require 'recurrent'
----------------------------------------------------------------------
-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.model .. '.lua ' .. opt.save)

cmd:log(opt.save .. '/log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local optStateFilename = paths.concat(opt.save,'optState')

local vocabSize = data.vocabSize
local vocab = data.vocab
local decoder = data.decoder
local decode = data.decode


local trainRegime = modelConfig.regime
local recurrentEncoder = modelConfig.recurrentEncoder
local recurrentDecoder = modelConfig.recurrentDecoder
local embedder = modelConfig.embedder
local classifier = modelConfig.classifier

-- Model + Loss:

local criterion = nn.CrossEntropyCriterion()--n.ClassNLLCriterion()


local TensorType = 'torch.FloatTensor'
local model = nn.Sequential():add(embedder):add(recurrentEncoder):add(classifier):add(recurrentDecoder)

if opt.type =='cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
    model:cuda()
    criterion = criterion:cuda()
    TensorType = 'torch.CudaTensor'
    initState = nn.utils.recursiveType(initState, TensorType)


    ---Support for multiple GPUs - currently data parallel scheme
    if opt.nGPU > 1 then
        initState:resize(opt.batchSize / opt.nGPU, stateSize)
        local net = model
        model = nn.DataParallelTable(1)
        model:add(net, 1)
        for i = 2, opt.nGPU do
            cutorch.setDevice(i)
            model:add(net:clone(), i)  -- Use the ith GPU
        end
        cutorch.setDevice(opt.devid)
    end
end

--sequential criterion
local seqCriterion = nn.TemporalCriterion(criterion, true)

-- Optimization configuration
local Weights,Gradients = model:getParameters()

local savedModel = {
    embedder = embedder:clone('weight','bias', 'running_mean', 'running_std'),
    recurrentEncoder = recurrentEncoder:clone('weight','bias', 'running_mean', 'running_std'),
    recurrentDecoder = recurrentDecoder:clone('weight','bias', 'running_mean', 'running_std'),
    classifier = classifier:clone('weight','bias', 'running_mean', 'running_std')
}

----------------------------------------------------------------------
print '\n==> Network'
print(model)
print('\n==>' .. Weights:nElement() ..  ' Parameters')

print '\n==> Criterion'
print(criterion)

if trainRegime then
    print '\n==> Training Regime'
    table.foreach(trainRegime, function(x, val) print(string.format('%012s',x), unpack(val)) end)
end
------------------Optimization Configuration--------------------------

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}



----------------------------------------------------------------------
---utility functions

local function reshapeData(wordVec, seqLength, batchSize)
    local offset = offset or 0
    local length = wordVec:nElement()
    local numBatches = torch.floor(length / (batchSize * seqLength))

    local batchWordVec = wordVec.new():resize(numBatches, batchSize, seqLength)
    local endWords = wordVec.new():resize(numBatches, batchSize, 1)

    local endIdxs = torch.LongTensor()
    for i=1, batchSize do
        local startPos = torch.round((i - 1) * length / batchSize ) + 1
        local sliceLength = seqLength * numBatches
        local endPos = startPos + sliceLength - 1

        batchWordVec:select(2,i):copy(wordVec:narrow(1, startPos, sliceLength))
        endIdxs:range(startPos + seqLength, endPos + 1, seqLength)
        endWords:select(2,i):copy(wordVec:index(1, endIdxs))
    end
    return batchWordVec, endWords
end

local function saveModel(epoch)
    local fn = netFilename .. '_' .. epoch .. '.t7'
    torch.save(fn,
    {
        embedder = savedModel.embedder:clone():float(),
        recurrentEncoder = savedModel.recurrentEncoder:clone():float(),
        recurrentDecoder = savedModel.recurrentEncoder:clone():float(),
        classifier = savedModel.classifier:clone():float(),
        inputSize = inputSize,
        stateSize = stateSize,
        vocab = vocab,
        decoder = decoder
    })
    collectgarbage()
end

----------------------------------------------------------------------
local function ForwardSeq(dataVec, train)

    local data, labels = reshapeData(dataVec, opt.seqLength, opt.batchSize )
    local sizeData = data:size(1)
    local numSamples = 0
    local lossVal = 0
    local currLoss = 0
    local NewSentenceVal = vocab['<NS>']
    local x = torch.Tensor(opt.batchSize, opt.seqLength + 2):type(TensorType):fill(NewSentenceVal)
    local yt = torch.Tensor():type(TensorType)
    local encX

    -- input is a sequence
    recurrentEncoder:sequence()
    recurrentDecoder:sequence()

    recurrentEncoder:forget()
    recurrentDecoder:forget()


    local encoderModel = nn.Sequential()
    encoderModel:add(nn.Narrow(2, 2, opt.seqLength))
    encoderModel:add(nn.Reverse(2):type(TensorType))--reversing the sequence for better training
    encoderModel:add(recurrentEncoder)

    local decoderModel = nn.Sequential()
    decoderModel:add(nn.Narrow(2, 1, opt.seqLength + 1))
    decoderModel:add(recurrentDecoder)
    --decoderModel:add(nn.Narrow(2, 1, opt.seqLength))
    decoderModel:add(nn.View(opt.batchSize * (opt.seqLength + 1), -1))
    decoderModel:add(classifier)
    decoderModel:add(nn.View(opt.batchSize, opt.seqLength + 1, -1))

    for b=1, sizeData do
        recurrentEncoder:zeroState()
        recurrentDecoder:zeroState()

        x:narrow(2, 2, opt.seqLength):copy(data[b])
        local embX = embedder:forward(x)
        local target = x:narrow(2, 2, opt.seqLength + 1)
        yt:resizeAs(target):copy(target)

        local res = encoderModel:forward(embX)
        recurrentDecoder:setState(recurrentEncoder:getState())
        local y = decoderModel:forward(embX)
        currLoss = seqCriterion:forward(y,yt)
      --  print(currLoss/opt.seqLength)
        if train then
          local f_eval = function()

             Gradients:zero()
          --   local dE_dyt = 0
            local dE_dy = seqCriterion:backward(y,yt)
            local dE_dembX1 = decoderModel:backward(embX, dE_dy)
            recurrentEncoder:accGradState(recurrentDecoder:getGradState())

            local dE_dembX2 = encoderModel:backward(embX, res:zero())
            embedder:backward(x, dE_dembX1 + dE_dembX2)
            local norm = Gradients:norm()
            if norm > 5 then
              local shrink = 5 / norm
              Gradients:mul(shrink)
            end
          --  Gradients:clamp(-5,5)
            return currLoss, Gradients
          end

            if opt.nGPU > 1 then
                model:syncParameters()
            end
            _G.optim[opt.optimization](f_eval, Weights, optimState)

        end
        lossVal = currLoss / opt.seqLength + lossVal
        numSamples = numSamples + x:size(1)
        xlua.progress(numSamples, sizeData*opt.batchSize)
    end

    collectgarbage()
    xlua.progress(numSamples, sizeData)
    return lossVal / sizeData
end


----------------------------------------------------------------------
local function seq2seq(encoderTbl, ...)
  -- input is of form {rnn, data}, {rnn,data},...
  -- data is ordered batches X batchSize X smpLength

    local inputData, encoderModel = unpack(encoderTbl)
    local xE = torch.Tensor():type(TensorType)

    local decoderTbls = {...}
    local outputData = {}
    local decoderModels = {}
    local xDs = {}
    local targets = {}
    for num, decTbl in pairs(decoderTbls) do
      outputData[num], decoderModels[num] = unpack(decTbl)
      xDs[num] = torch.Tensor():type(TensorType)
      targets[num] = torch.Tensor():type(TensorType)
    end
    local numBatches = inputData:size(1)
    local numSamples = 0
    local lossVals = torch.FloatTensor(#decoderModels):zero()

    applyRecurrent({encoderModel, unpack(decoderModels)}, 'sequence')
    applyRecurrent({encoderModel, unpack(decoderModels)}, 'forget')

    for b = 1, numBatches do
        applyRecurrent({encoderModel, unpack(decoderModels)}, 'zeroState')

        x:resizeAs(inputData[b]):copy(inputData[b])
        local out = encoderModel:forward(x)
        local state = encoderModel:getState()

        applyRecurrent({unpack(decoderModels)}, 'setState', state)
        for m = 1, #decoderModels do
          xDs[m]:resizeAs(outputData[m][b]):copy(outputData[m][b])
          targets[m]:resizeAs(xDs[m]:sub(1,-1,2,-1)):copy(xDs[m]:sub(1,-1,2,-1))
          local y = decoderModels[m]:forward(x)
          local currLoss = seqCriterion:forward(y, targets[m])
          lossVals[m] = lossVals[m] + currLoss / opt.seqLength
        end
        ----Training -> backpropagation
        if train then
            local f_eval = function()
              Gradients:zero()
              for m = 1, #decoderModels do
                local dE_dy = seqCriterion:backward(y,yt)
                decoderModels[m]:backward(x, dE_dy)
                local decoderGradState = applyRecurrent(decoderModel, 'getGradState')
                local encoderGradState = applyRecurrent(decoderModel, 'getGradState')
                applyRecurrent(encoderModel, 'setGradState', decoderGradState + encoderGradState)
              end
              encoderModel:backward(x, out:zero())
              --Gradient clipping (actually normalizing)
              local norm = Gradients:norm()
              if norm > 5 then
                local shrink = 5 / norm
                Gradients:mul(shrink)
              end
              return currLoss, Gradients
            end

            if opt.nGPU > 1 then
                model:syncParameters()
            end
            _G.optim[opt.optimization](f_eval, Weights, optimState)
        end
        numSamples = numSamples + x:size(1)
        xlua.progress(numSamples, numBatches * opt.batchSize)
    end

    collectgarbage()
    xlua.progress(numSamples, numBatches * opt.batchSize)
    lossVals:div(numBatches)
    return lossVals
end
------------------------------

local function train(dataVec)
    model:training()
    return ForwardSeq(dataVec, true)
end

local function evaluate(dataVec)
    model:evaluate()
    return ForwardSeq(dataVec, false)
end

local function sample(str, num, space, temperature)
    local num = #str:split(' ')
    local temperature = 0-- temperature or 1
    local function smp(preds)
        if temperature == 0 then
            local _, num = preds:max(2)
            return num[1]
        else
            preds = nn.SoftMax():type(TensorType):forward(preds)
            preds:div(temperature) -- scale by temperature
            local probs = preds:squeeze()
            probs:div(probs:sum()) -- renormalize so probs sum to one
            local num = torch.multinomial(probs:float(), 1):typeAs(preds)
            return num
        end
    end
    --local embedder = savedModel.embedder
    --local recurrentEncoder = savedModel.recurrentEncoder
    --local recurrentDecoder = savedModel.recurrentDecoder
    --local classifier = savedModel.classifier

    recurrentEncoder:evaluate()
    recurrentDecoder:evaluate()

    recurrentEncoder:sequence()
    recurrentDecoder:single()
    local stateModel = nn.Sequential():add(embedder):add(nn.Reverse(2):type(TensorType)):add(recurrentEncoder)

    local sentState
    local pred, predText, embedded
    predText = ' '
    local encoded = data.encode(str)
    print('\nOriginal:\n' .. decode(encoded))
    stateModel:forward(encoded:view(1, -1))
    recurrentDecoder:setState(recurrentEncoder:getState())
    wordNum = data.encode('<NS>')
    for i=1, num do
      print(recurrentDecoder:getState()[1]:mean())
        sentState = recurrentDecoder:forward(embedder:forward(wordNum))
        pred = classifier:forward(sentState)
        wordNum = smp(pred)
        predText = predText .. ' ' .. decoder[wordNum:squeeze()]
    end
    return predText
end
return {
    train = train,
    evaluate = evaluate,
    sample = sample,
    saveModel = saveModel,
    optimState = optimState,
    model = model
}
