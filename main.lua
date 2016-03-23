require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
require 'eladtools'
require 'seqProvider'
require 'Seq2Seq'
-------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training recurrent networks on word-level text dataset')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Data Options')
cmd:option('-shuffle',            false,                       'shuffle training samples')

cmd:text('===>Model And Training Regime')
cmd:option('-model',              'GRU',                      'Recurrent model [RNN, iRNN, LSTM, GRU]')
cmd:option('-seqLength',          50,                          'number of timesteps to unroll for')
cmd:option('-rnnSize',            256,                         'size of rnn hidden layer')
cmd:option('-embeddingSize',      256,                          'size of word embedding')
cmd:option('-numLayers',          1,                           'number of layers in the LSTM')
cmd:option('-dropout',            0,                           'dropout p value')
cmd:option('-LR',                 1e-3,                        'learning rate')
cmd:option('-LRDecay',            0,                           'learning rate decay (in # samples)')
cmd:option('-weightDecay',        0,                           'L2 penalty on the weights')
cmd:option('-momentum',           0,                           'momentum')
cmd:option('-batchSize',          32,                          'batch size')
cmd:option('-decayRate',          2,                           'exponential decay rate')
cmd:option('-initWeight',         0.05,                        'uniform weight initialization range')
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

if opt.type =='cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
    cutorch.setHeapTracking(true)
end

local types = {
  cuda = 'torch.CudaTensor',
  float = 'torch.FloatTensor',
  cl = 'torch.ClTensor',
  double = 'torch.DoubleTensor'
}

local TensorType = types[opt.type] or 'torch.FloatTensor'
-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'LossRate.log')
local log = optim.Logger(logFilename)

----------------------------------------------------------------------
-- Model + Loss:
local encodedSeq = seqProvider{
  source = '/home/ehoffer/Datasets/Translation/news-commentary-v10.fr-en.en', maxLength = opt.seqLength, limitVocab = 50000,
  type = TensorType, preprocess = false
}
local decodedSeq = seqProvider{
  source = '/home/ehoffer/Datasets/Translation/news-commentary-v10.fr-en.fr', maxLength = opt.seqLength, type = TensorType, limitVocab = 50000,
  padStart = true, padEnd = true, preprocess = false
}

print(decodedSeq:decode(decodedSeq:getBatch(1)[1]))
print('Encoded Vocabulary:', encodedSeq.vocab:size())
print('Decoded Vocabulary:', decodedSeq.vocab:size())
--print('Dataset size (encoded=decoded)', encodedSeq:size(), decodedSeq:size())
print('Dataset size (encoded=decoded)', encodedSeq:size(), decodedSeq:size())
if paths.filep(opt.load) then
    local modelConfig = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
    recurrentEncoder = modelConfig.recurrentEncoder
    recurrentDecoder = modelConfig.recurrentDecoder
    classifier = modelConfig.classifier
else
    local rnnTypes = {LSTM = nn.LSTM, RNN = nn.RNN, GRU = nn.GRU, iRNN = nn.iRNN}
    local rnn = rnnTypes[opt.model]
    local hiddenSize = opt.embeddingSize
    recurrentEncoder = nn.Sequential():add(nn.LookupTable(encodedSeq.vocab:size(), opt.embeddingSize, encodedSeq.vocab:pad())):add(nn.Reverse(2))
    for i=1, opt.numLayers do
        recurrentEncoder:add(rnn(hiddenSize, opt.rnnSize, opt.initWeight))
        if opt.dropout > 0 then
            recurrentEncoder:add(nn.Dropout(opt.dropout))
        end
        hiddenSize = opt.rnnSize
    end

    hiddenSize = opt.embeddingSize
    recurrentDecoder = nn.Sequential():add(nn.LookupTable(decodedSeq.vocab:size(), opt.embeddingSize, decodedSeq.vocab:pad()))
    for i=1, opt.numLayers do
        recurrentDecoder:add(rnn(hiddenSize, opt.rnnSize, opt.initWeight))
        if opt.dropout > 0 then
            recurrentDecoder:add(nn.Dropout(opt.dropout))
        end
        hiddenSize = opt.rnnSize
    end
    classifier = nn.Linear(opt.rnnSize, decodedSeq.vocab:size())
end

classifier:share(recurrentDecoder:get(1), 'weight','gradWeight')

--local lookupZero = function(self, input) --used to ensure padding is a zeros vector
--
--  local out = nn.LookupTable.updateOutput(self, input)
--  local dim =  input:dim() + 1
--  out:cmul(nn.utils.addSingletonDimension(input:ne(0),dim):expandAs(out))
----  embedder.weight[vocab['<PAD>']]:zero()
----  return nn.LookupTable.updateOutput(...)
--return out
--end
--recurrentEncoder:get(1).updateOutput = lookupZero
--recurrentDecoder:get(1).updateOutput = lookupZero
recurrentEncoder:get(1).weight[encodedSeq.vocab:pad()]:zero()
recurrentDecoder:get(1).weight[decodedSeq.vocab:pad()]:zero()
local criterion = nn.CrossEntropyCriterion():type(TensorType)






local enc = nn.Sequential()
enc:add(recurrentEncoder)

local dec = nn.Sequential()
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



local seq2seq = Seq2Seq{
  encoder = enc,
  optimFunc=_G.optim[opt.optimization],
  optimState = optimState,
  criterion = criterion,
  batchSize = opt.batchSize,
  padValue = decodedSeq.vocab:pad()
}
seq2seq:addDecoder(dec)
seq2seq:type(TensorType)
seq2seq:flattenParameters()

----------------------------------------------------------------------
print '\n==> Encoder'
print(enc)

print '\n==> Decoder'
print(dec)

print '\n==> Criterion'
print(criterion)


----------------------------------------------------------------------
---utility functions
function saveModel(epoch)
  local fn = netFilename .. '_' .. epoch .. '.t7'
  torch.save(fn,
  {
    recurrentEncoder = recurrentEncoder:clearState(),
    recurrentDecoder = recurrentDecoder:clearState(),
    classifier = classifier:clearState(),
    vocabEncode = encodedSeq.vocab,
    vocabDecode = decodedSeq.vocab
  })
  collectgarbage()
end



function sample(encoderTbl, decoderTbl, str)
    local num = #str:split(' ')
    local encoderModel, encodeVocab = unpack(encoderTbl)
    local decoderModel, decodeVocab = unpack(decoderTbl)
    encoderModel:zeroState()
    decoderModel:single()
    encoderModel:sequence()
    encoderModel:evaluate()
    decoderModel:evaluate()
    local pred, predText, embedded
    predText = {}
    local encoded = encodeVocab:encode(str):type(TensorType)
    print('\nOriginal:\n' .. table.concat(encodeVocab:decode(encoded) ,' '))
    wordNum = torch.Tensor({decodeVocab:go()}):type(TensorType)
    if encoded:nElement() == 1 then
      encoderModel:single()
    end
    encoderModel:forward(encoded:view(1, -1))
    decoderModel:setState(encoderModel:getState())
    for i=1, num do
        local pred = decoderModel:forward(wordNum)
        pred:select(2, decodeVocab:unk()):zero()
        _, wordNum = pred:max(2)
        wordNum = wordNum[1]
        if wordNum:squeeze() == decodeVocab:eos() then
          break
        end
        table.insert(predText, decodeVocab:decode(wordNum)[1])
    end
    return table.concat(predText, ' ')
end


local sampledDec = nn.Sequential():add(recurrentDecoder):add(classifier):add(nn.SoftMax():type(TensorType))

local decreaseLR = EarlyStop(1,opt.epochDecay)
local stopTraining = EarlyStop(opt.earlyStop, opt.epoch)
local epoch = 1



repeat

    print('\nEpoch ' .. epoch ..'\n')
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'Lately, with gold prices up more than 300% over the last decade, it is harder than ever.'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'a form of asbestos once used to make kent cigarette filters has caused a high percentage of cancer deaths among a group of workers'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},"the following were among yesterday 's offerings and pricings in the u.s. and non-u.s. capital markets"))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'nothing could be further from the truth'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'Are you one of millions out there who are trying to learn foreign language, but never have enough time?'))
    --local LossTrain = seq2seq({enc, encodedSeq},{{dec, decodedSeq}}, true)
      local LossTrain = seq2seq:learn(encodedSeq,{decodedSeq})
    --local LossTrain = seq2seq({enc, encodedSeq},{{dec1, decodedSeq}, {dec2, decodedSeq}}, true)
    saveModel(epoch)

    if opt.optState then
        torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
    end
    print('\nTraining Perplexity: ' .. torch.exp(LossTrain))

    --local LossVal = evaluate(data.validationData)

    --print('\nValidation Perplexity: ' .. torch.exp(LossVal))

    --local LossTest = evaluate(data.testData)


    --print('\nTest Perplexity: ' .. torch.exp(LossTest))
    --log:add{['Training Loss']= LossTrain, ['Validation Loss'] = LossVal, ['Test Loss'] = LossTest}
    --log:style{['Training Loss'] = '-', ['Validation Loss'] = '-', ['Test Loss'] = '-'}
    --log:plot()
    epoch = epoch + 1

    --if decreaseLR:update(LossVal) then
    --    optimState.learningRate = optimState.learningRate / opt.decayRate
    --    print("Learning Rate decreased to: " .. optimState.learningRate)
    --    decreaseLR = EarlyStop(1,1)
    --    decreaseLR:reset()
    --end

until false--stopTraining:update(LossVal)

local lowestLoss, bestIteration = stopTraining:lowest()

print("Best Iteration was " .. bestIteration .. ", With a validation loss of: " .. lowestLoss)
