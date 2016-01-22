require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
require 'eladtools'
local tds = require 'tds'
-------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training recurrent networks on word-level text dataset - Penn Treebank')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Data Options')
cmd:option('-shuffle',            false,                       'shuffle training samples')

cmd:text('===>Model And Training Regime')
cmd:option('-model',              'LSTM',                      'Recurrent model [RNN, iRNN, LSTM, GRU]')
cmd:option('-seqLength',          50,                          'number of timesteps to unroll for')
cmd:option('-rnnSize',            128,                         'size of rnn hidden layer')
cmd:option('-numLayers',          1,                           'number of layers in the LSTM')
cmd:option('-dropout',            0,                           'dropout p value')
cmd:option('-LR',                 1e-3,                        'learning rate')
cmd:option('-LRDecay',            0,                           'learning rate decay (in # samples)')
cmd:option('-weightDecay',        0,                           'L2 penalty on the weights')
cmd:option('-momentum',           0,                           'momentum')
cmd:option('-batchSize',          32,                          'batch size')
cmd:option('-decayRate',          2,                           'exponential decay rate')
cmd:option('-initWeight',         0.08,                        'uniform weight initialization range')
cmd:option('-earlyStop',          5,                           'number of bad epochs to stop after')
cmd:option('-optimization',       'adam',                   'optimization method')
cmd:option('-gradClip',           5,                           'clip gradients at this value')
cmd:option('-epoch',              100,                         'number of epochs to train')
cmd:option('-epochDecay',         5,                           'number of epochs to start decay learning rate')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                           'number of threads')
cmd:option('-type',               'cuda',                      'float or cuda')
cmd:option('-devid',              1,                           'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                           'num of gpu devices used')
cmd:option('-seed',               123,                         'torch manual random number generator seed')
cmd:option('-constBatchSize',     false,                       'do not allow varying batch sizes')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                          'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''),      'save directory')
cmd:option('-optState',           false,                       'Save optimization state every epoch')
cmd:option('-checkpoint',         0,                           'Save a weight check point every n samples. 0 for off')




opt = cmd:parse(arg or {})
opt.save = paths.concat('./Results', opt.save)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'LossRate.log')
local log = optim.Logger(logFilename)
----------------------------------------------------------------------
local data = torch.load('./cache/ptb_cached.t7')
local decoder = data.decodeTable
local vocab = data.vocab
local decode = data.decodeFunc
local vocabSize = #decoder
----------------------------------------------------------------------
-- Model + Loss:

if paths.filep(opt.load) then
    local modelConfig = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
else
    local rnnTypes = {LSTM = nn.LSTM, RNN = nn.RNN, GRU = nn.GRU, iRNN = nn.iRNN}
    local rnn = rnnTypes[opt.model]
    local hiddenSize = vocabSize
    recurrentEncoder = nn.Sequential()
    for i=1, opt.numLayers do
        recurrentEncoder:add(rnn(hiddenSize, opt.rnnSize, opt.initWeight))
        if opt.dropout > 0 then
            recurrentEncoder:add(nn.Dropout(opt.dropout))
        end
        hiddenSize = opt.rnnSize
    end
    recurrentDecoder = recurrentEncoder:clone()
    recurrentDecoder:reset()
    require 'utils.OneHot'
    embedder = nn.OneHot(vocabSize, vocab['<PAD>'])--nn.LookupTable(vocabSize, vocabSize)
    classifier = nn.Linear(opt.rnnSize, vocabSize)
end

local criterion = nn.CrossEntropyCriterion()

local allModules = nn.Sequential():add(embedder):add(recurrentDecoder):add(recurrentEncoder):add(classifier)

local TensorType = 'torch.FloatTensor'

if opt.type =='cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
    cutorch.setHeapTracking(true)
    criterion = criterion:cuda()
    allModules:cuda()
    TensorType = 'torch.CudaTensor'
end

local weights, gradients = allModules:getParameters()


local enc = nn.Sequential()
enc:add(embedder)
enc:add(nn.Reverse(2):type(TensorType))
enc:add(recurrentEncoder)

local dec = nn.Sequential()
dec:add(embedder:clone('weight','gradWeight'))
dec:add(recurrentDecoder)
dec:add(nn.TemporalModule(classifier))


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
print '\n==> Encoder'
print(enc)

print '\n==> Decoder'
print(dec)
print('\n==>' .. weights:nElement() ..  ' Parameters')

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
    recurrentDecoder = savedModel.recurrentDecoder:clone():float(),
    classifier = savedModel.classifier:clone():float(),
    vocab = vocab,
    decoder = decoder
  })
  collectgarbage()
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
  function seq2seq(encoderTbl, decoderTbls, train)
    -- input is of form {data, model, vocab}, {data, model, vocab},...
    -- data is ordered batches X batchSize X smpLength
    local maxLength = 50
    local inputData, encoderModel, encoderVocab = unpack(encoderTbl)
    local xE = torch.Tensor(opt.batchSize, maxLength + 2):type(TensorType)
    if train then
      encoderModel:training()
    else
      encoderModel:evaluate()
    end
    encoderModel:sequence()
    encoderModel:forget()
    encoderModel:setIterations(maxLength)
    local outputData = {}
    local decoderModels = {}
    local decoderVocabs = {}
    local xDs = {}
    local targets = {}

    for num, decTbl in pairs(decoderTbls) do
      outputData[num], decoderModels[num], decoderVocabs[num] = unpack(decTbl)
      if train then
        decoderModels[num]:training()
      else
        decoderModels[num]:evaluate()
      end
      decoderModels[num]:sequence()
      decoderModels[num]:forget()
      decoderModels[num]:setIterations(maxLength)
      xDs[num] = torch.Tensor(opt.batchSize , maxLength + 1):type(TensorType)
    end
    local numBatches = math.floor(#inputData / opt.batchSize)
    local numSamples = 1
    local lossVals = torch.FloatTensor(#decoderModels):zero()
    local randIndexes = torch.LongTensor():randperm(#inputData)

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
        local idx = randIndexes[i + numSamples - 1]
        local currSeq = data[idx]
        local currLength = currSeq:nElement()
        currMaxLength = math.max(currMaxLength, currLength)
        if numData == 0 then --encoder
          x[{i,{1, currLength}}]:copy(currSeq)
        else
          x[{i,{2, currLength + 1}}]:copy(currSeq)
          x[{i, currLength + 2}] = eosVal
          x[{i, 1}] = goVal
        end
      end
      local target
      if numData == 0 then --encoder
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
  --     print(torch.exp(currLoss))
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
            dE_dy:cmul(yt[m]:ne(vocab['<PAD>']):view(yt[m]:size(1),yt[m]:size(2), 1):expandAs(dE_dy))
            decoderModels[m]:backward(x[m], dE_dy)
            local decoderGradState = decoderModels[m]:getGradState()
            encoderModel:accGradState(decoderGradState)
          end
          encoderModel:backward(x[0], out:zero())
          --Gradient clipping (actually normalizing)
          local norm = gradients:norm()
          if norm > 5 then
            local shrink = 5 / norm
            gradients:mul(shrink)
          end
          return currLoss, gradients
        end

        _G.optim[opt.optimization](f_eval, weights, optimState)
      end
      numSamples = numSamples + opt.batchSize

      xlua.progress(numSamples - 1, #inputData)
    end

    collectgarbage()
    xlua.progress(#inputData, #inputData)
    lossVals:div(numBatches)
    return lossVals:mean()
  end
------------------------------


function sample(encoderTbl, decoderTbl, str)
    local num = 50--#str--:split(' ')
    local encoderModel, encodeFunc = unpack(encoderTbl)
    local decoderModel, decodeFunc = unpack(decoderTbl)
    local padToken = vocab['<PAD>']
    encoderModel:zeroState()
    decoderModel:single()
    encoderModel:sequence()
    encoderModel:evaluate()
    decoderModel:evaluate()
    local pred, predText, embedded
    predText = ''
    local encoded = encodeFunc(str):type(TensorType)
    print('\nOriginal:\n' .. decodeFunc(encoded))
    wordNum = torch.Tensor({vocab['<GO>']}):type(TensorType)
    if encoded:nElement() == 1 then
      encoderModel:single()
    end
    encoderModel:forward(encoded:view(1, -1))
    decoderModel:setState(encoderModel:getState())
    for i=1, num do
        local pred = decoderModel:forward(wordNum)
        pred:select(2, padToken):zero()
        _, wordNum = pred:max(2)
        wordNum = wordNum[1]
        if wordNum:squeeze() == vocab['<EOS>'] then
          break
        end
        predText = predText .. ' ' .. decodeFunc(wordNum)
    end
    return predText
end


local sampledDec = nn.Sequential():add(embedder):add(recurrentDecoder):add(classifier):add(nn.SoftMax():type(TensorType))

local decreaseLR = EarlyStop(1,opt.epochDecay)
local stopTraining = EarlyStop(opt.earlyStop, opt.epoch)
local epoch = 1
repeat
    print('\nEpoch ' .. epoch ..'\n')
    print('\nSampled Text:\n' .. sample({enc, data.encodeFunc}, {sampledDec,data.decodeFunc},'a'))
    print('\nSampled Text:\n' .. sample({enc, data.encodeFunc}, {sampledDec,data.decodeFunc},'b'))
    print('\nSampled Text:\n' .. sample({enc, data.encodeFunc}, {sampledDec,data.decodeFunc},'once'))
    print('\nSampled Text:\n' .. sample({enc, data.encodeFunc}, {sampledDec,data.decodeFunc},'cigarette'))
    print('\nSampled Text:\n' .. sample({enc, data.encodeFunc}, {sampledDec,data.decodeFunc},"followings"))
    local LossTrain = seq2seq({data.sentences, enc, data.vocab},{{data.sentences, dec, data.vocab}}, true)
    saveModel(epoch)
    if opt.optState then
        torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
    end
    print('\nTraining Perplexity: ' .. torch.exp(LossTrain))

    epoch = epoch + 1

until false

local lowestLoss, bestIteration = stopTraining:lowest()

print("Best Iteration was " .. bestIteration .. ", With a validation loss of: " .. lowestLoss)