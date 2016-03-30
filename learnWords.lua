require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
require 'eladtools'
require 'seqProvider'
require 'Seq2Seq'
-------------------------------------------------------

dofile('opt.lua')
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'LossRate.log')
local log = optim.Logger(logFilename)

----------------------------------------------------------------------
local wordsFile = 'ptb_words.txt'
local words = vocabulary{source='../recurrent.torch/examples/language/data/ptb.train.txt', tokenizer = function(l) return l:split(' ') end}
print(words:size())
words:saveAsText(wordsFile)

local function splitChars(str)
  local function split(str)
    if #str>0 then
      return str:sub(1,1), split(str:sub(2))
    end
  end
  return {split(str)}
end


local encodedSeq = seqProvider{
  source = wordsFile, maxLength = opt.seqLength, type = opt.tensorType,
  preprocess = true, tokenizer = splitChars
}
print('Encoded Vocabulary:', encodedSeq.vocab:size())

encodedSeq.vocab:saveAsText('check.txt')
local decodedSeq = seqProvider{
  source = wordsFile, maxLength = opt.seqLength, type = opt.tensorType,
  padStart = true, padEnd = true, preprocess = true, vocab = encodedSeq.vocab
}
print(encodedSeq:encode('al'))
print('Decoded Vocabulary:', decodedSeq.vocab:size())

print('Dataset size (encoded=decoded)', encodedSeq:size(), decodedSeq:size())
if paths.filep(opt.load) then
    local modelConfig = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
    recurrentEncoder = modelConfig.recurrentEncoder
    recurrentDecoder = modelConfig.recurrentDecoder
    classifier = modelConfig.classifier
else
    recurrentEncoder = buildBiDiRNN(encodedSeq.vocab:size())
    recurrentDecoder = buildRNN(decodedSeq.vocab:size(), false)
    classifier = nn.Linear(opt.rnnSize, decodedSeq.vocab:size())
end
recurrentEncoder:sequence()
recurrentEncoder:forward(torch.ones(16,5))
function getEmbedder(m)
  return m:findModules('nn.LookupTable')[1]
end
--tie classification and embedding weights
getEmbedder(recurrentEncoder):share(getEmbedder(recurrentDecoder), 'weight','gradWeight')
classifier:share(getEmbedder(recurrentDecoder), 'weight','gradWeight')

function zeroPadWeights(m, padIdx)
  classifier.weight[padIdx]:zero()
  classifier.bias[padIdx] = 0
end

zeroPadWeights(recurrentDecoder, encodedSeq.vocab:pad())
local criterion = nn.CrossEntropyCriterion():type(opt.tensorType)


local enc = recurrentEncoder
local dec = nn.Sequential()
dec:add(recurrentDecoder)
dec:add(nn.TemporalModule(classifier))


local criterion = nn.CrossEntropyCriterion()


local enc = nn.Sequential()
enc:add(recurrentEncoder)

local dec = nn.Sequential()
dec:add(recurrentDecoder)
dec:add(nn.TemporalModule(classifier))
print(enc)

-- Optimization configuration
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}

local seq2seq = Seq2Seq{
  encoder = enc,
  optimFunc =_G.optim[opt.optimization],
  optimState = optimState,
  criterion = criterion,
  batchSize = opt.batchSize
}
seq2seq:addDecoder(dec)
seq2seq:type(opt.tensorType)
seq2seq:flattenParameters()




----------------------------------------------------------------------
print '\n==> Encoder'
print(enc)

print '\n==> Decoder'
print(dec)
print('\n==>' .. seq2seq.params[1]:nElement() ..  ' Parameters')

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



------------------------------


function sample(encoderTbl, decoderTbl, str)
    local num = 50--opt.seqLength
    local encoderModel, encodeVocab = unpack(encoderTbl)
    local decoderModel, decodeVocab = unpack(decoderTbl)
    encoderModel:zeroState()
    decoderModel:single()
    encoderModel:sequence()
    encoderModel:evaluate()
    decoderModel:evaluate()
    local pred, predText, embedded
    predText = {}
    local encoded = encodeVocab:encode(str):type(opt.tensorType)

    print('\nOriginal:\n' .. table.concat(encodeVocab:decode(encoded),' '))
    wordNum = torch.Tensor({decodeVocab:go()}):type(opt.tensorType)
    --if encoded:nElement() == 1 then
    --  encoded = encodeVocab:encode(str .. 'a'):type(opt.tensorType)
    --  encoded[encoded:size(1)]=0
    --end
--  if encoded:size(1) then encoderModel:sequence() end
    encoderModel:forward(encoded:view(1, -1))
    decoderModel:setState(encoderModel:getState())
    for i=1, num do
      --print(wordNum)
        local pred = decoderModel:forward(wordNum)
        pred:select(2, decodeVocab:unk()):zero()
        _, wordNum = pred:max(2)
        wordNum = wordNum[1]
        if wordNum:squeeze() == decodeVocab:eos() then
          break
        end
        table.insert(predText, decodeVocab:decode(wordNum)[1])
    end
    return table.concat(predText,'')
end


local sampledDec = nn.Sequential():add(recurrentDecoder):add(classifier):add(nn.SoftMax():type(opt.tensorType))

local decreaseLR = EarlyStop(1,opt.epochDecay)
local stopTraining = EarlyStop(opt.earlyStop, opt.epoch)
local epoch = 1

repeat
    print('\nEpoch ' .. epoch ..'\n')

    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'a'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'b'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'nothing'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'millions'))
    print(getEmbedder(recurrentDecoder).weight[encodedSeq.vocab:pad()]:clone():abs():sum())
    local LossTrain = seq2seq:learn(encodedSeq,{decodedSeq})
    saveModel(epoch)

    if opt.optState then
        torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
    end
    print('\nTraining Perplexity: ' .. torch.exp(LossTrain))

    epoch = epoch + 1


until false--stopTraining:update(LossVal)

local lowestLoss, bestIteration = stopTraining:lowest()

print("Best Iteration was " .. bestIteration .. ", With a validation loss of: " .. lowestLoss)
