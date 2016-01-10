
----------------------------------------------------------------------
-- Output files configuration
os.execute('mkdir -p ' .. opt.save)

cmd:log(opt.save .. '/log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')

local vocabSize = data.vocabSize
local vocab = data.vocab
local decoder = data.decodeTable


local recurrentEncoder = modelConfig.recurrentEncoder
local recurrentDecoder = modelConfig.recurrentDecoder
local embedder = modelConfig.embedder
local classifier = modelConfig.classifier

-- Model + Loss:

local criterion = nn.CrossEntropyCriterion()


local TensorType = 'torch.FloatTensor'

if opt.type =='cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.devid)
  cutorch.manualSeed(opt.seed)
  --model:cuda()
  criterion = criterion:cuda()
  TensorType = 'torch.CudaTensor'
end

--sequential criterion
local seqCriterion = nn.TemporalCriterion(criterion)

-- Optimization configuration
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}


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


----------------------------------------------------------------------
---utility functions

function saveModel(epoch)
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

local function lossGradNoPadding(criterion, y, yt, padToken)
  local grad = y.new():resizeAs(y):zero()
  for i = 1, y:size(1) do
    for j = 1, y:size(2) do
      if yt[i][j] == padToken then break end
      grad[i][j]:copy(criterion:backward(y[i][j], yt[i][j]))
    end
  end
  grad:div(y:size(1))
  return grad
end

local function lossNoPadding(criterion, y, yt, padToken)
  local loss = 0
  for i = 1, y:size(1) do
    local currLength = 1
    local currLoss = 0
    for j = 1, y:size(2) do
      if yt[i][j] == padToken then break end
      currLoss = currLoss + criterion:forward(y[i][j], yt[i][j])
      currLength = currLength + 1
    end
    loss = loss + currLoss / currLength
  end
  loss = loss / y:size(1)
  return loss
end

----------------------------------------------------------------------
function seq2seq(encoderTbl, ...)
  -- input is of form {data, model, vocab}, {data, model, vocab},...
  -- data is ordered batches X batchSize X smpLength
  local maxLength = 50

  local inputData, encoderModel, encoderVocab = unpack(encoderTbl)
  local xE = torch.Tensor(opt.batchSize, maxLength):type(TensorType)
  encoderModel:training()
  encoderModel:sequence()
  encoderModel:forget()
  encoderModel:setIterations(maxLength)
  local decoderTbls = {...}
  local outputData = {}
  local decoderModels = {}
  local decoderVocabs = {}
  local xDs = {}
  local targets = {}

  for num, decTbl in pairs(decoderTbls) do
    outputData[num], decoderModels[num], decoderVocabs[num] = unpack(decTbl)
    decoderModels[num]:training()
    decoderModels[num]:sequence()
    decoderModels[num]:forget()
    decoderModels[num]:setIterations(maxLength)
    xDs[num] = torch.Tensor(opt.batchSize , maxLength + 2):type(TensorType)
  end
  local numBatches = math.floor(#inputData / opt.batchSize)
  local numSamples = 1
  local lossVals = torch.FloatTensor(#decoderModels):zero()

  function getNextBatch(numData) --0 for encoded data, otherwise num of decoded data
    local vocab = encoderVocab
    local data = inputData
    local x = xE
    if numData > 0 then
      data = outputData[numData]
      x = xDs[numData]
      vocab = decoderVocabs[numData]
    end
    local padVal = vocab['<PAD>']
    local eosVal = vocab['<EOS>']
    local goVal = vocab['<GO>']
    if numSamples + opt.batchSize > #data then
      return nil
    end
    x:fill(padVal)

    local currMaxLength = 0
    for i = 1, opt.batchSize do
      local currSeq = data[i + numSamples - 1]
      local currLength = currSeq:nElement()
      currMaxLength = math.max(currMaxLength, currLength)
      if numData == 0 then --decoder
        x[{i,{1, currLength}}]:copy(currSeq)
      else
        x[{i,{2, currLength + 1}}]:copy(currSeq)
        x[{i, currLength + 2}] = eosVal
        x[{i, 1}] = goVal
      end
    end
    local target
    if numData == 0 then --decoder
      x = x:narrow(2, 1, currMaxLength)
    else
      target = x:narrow(2, 2, currMaxLength + 1):contiguous()
      x = x:narrow(2, 1, currMaxLength + 1)
    end
    return x, target
  end


  for b = 1, numBatches do
    local x = {}
    local yt = {}
    local y = {}
    local currLoss = 0
    x[0] = getNextBatch(0)
    encoderModel:zeroState()
    local out = encoderModel:forward(x[0])

    local state = encoderModel:getState()

    for m = 1, #decoderModels do
      decoderModels[m]:setState(state)
      x[m], yt[m] = getNextBatch(m)

      y[m] = decoderModels[m]:forward(x[m])
      currLoss = lossNoPadding(criterion, y[m], yt[m], decoderVocabs[m]['<PAD>'])
    --  print(torch.exp(currLoss))
      lossVals[m] = lossVals[m] + currLoss --/ opt.seqLength
      seqCriterion:forward(y[m],yt[m])

    end
    ----Training -> backpropagation
    if train then
      local f_eval = function()
        encoderModel:zeroGradParameters()
        encoderModel:zeroGradState()

        for m = 1, #decoderModels do
          decoderModels[m]:zeroGradParameters()
          decoderModels[m]:zeroGradState()

          local dE_dy = seqCriterion:backward(y[m],yt[m])
          decoderModels[m]:backward(x[m], dE_dy)
          local decoderGradState = decoderModels[m]:getGradState()
          encoderModel:accGradState(decoderGradState)
        end
        encoderModel:backward(x[0], out:zero())
        --Gradient clipping (actually normalizing)
        local norm = Gradients:norm()
        if norm > 5 then
          local shrink = 5 / norm
          Gradients:mul(shrink)
        end
        return currLoss, Gradients
      end

      _G.optim[opt.optimization](f_eval, Weights, optimState)
    end
    numSamples = numSamples + opt.batchSize

    xlua.progress(numSamples, #inputData)
  end

  collectgarbage()
  xlua.progress(numSamples, #inputData)
  lossVals:div(numBatches)
  return lossVals:mean()
end
------------------------------

function train(dataVec)
  model:training()
  return ForwardSeq(dataVec, true)
end

function evaluate(dataVec)
  model:evaluate()
  return ForwardSeq(dataVec, false)
end

function sample(encoderTbl, decoderTbl, str)
  local num = #str:split(' ')
  local encoderModel, encodeFunc = unpack(encoderTbl)
  local decoderModel, decodeFunc = unpack(decoderTbl)
  local padToken = encodeFunc('<PAD>'):squeeze()
  encoderModel:zeroState()
  encoderModel:sequence()
  decoderModel:single()
  local sentState
  local pred, predText, embedded
  predText = ' '
  local encoded = encodeFunc(str)
  print('\nOriginal:\n' .. decodeFunc(encoded))
  encoderModel:forward(encoded:view(1, -1))
  wordNum = encodeFunc('<GO>')
  pred = decoderModel:forward(wordNum)
  decoderModel:setState(encoderModel:getState())
  for i=1, num do
    pred = decoderModel:forward(wordNum)
    pred:select(2, padToken):zero()
    _, wordNum = pred:max(2)
    wordNum = wordNum[1]
    predText = predText .. ' ' .. decodeFunc(wordNum)
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
