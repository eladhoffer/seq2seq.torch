require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
require 'eladtools'
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
cmd:option('-embeddingSize',      128,                          'size of word embedding')
cmd:option('-numLayers',          1,                           'number of layers in the LSTM')
cmd:option('-dropout',            0,                           'dropout p value')
cmd:option('-LR',                 1e-3,                        'learning rate')
cmd:option('-LRDecay',            0,                           'learning rate decay (in # samples)')
cmd:option('-weightDecay',        0,                           'L2 penalty on the weights')
cmd:option('-momentum',           0,                           'momentum')
cmd:option('-batchSize',          50,                          'batch size')
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

----------------------------------------------------------------------
local data = require 'data'
local decoder = data.decodeTable
local vocabSize = math.min(#decoder, 50000)
if vocabSize < #decoder then
  reduceNumWords(data.sentences, vocabSize, data.vocab['<UNK>'])
end
data.vocabSize = vocabSize
----------------------------------------------------------------------
if paths.filep(opt.load) then
    modelConfig = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
else
    modelConfig = {}
    local rnnTypes = {LSTM = nn.LSTM, RNN = nn.RNN, GRU = nn.GRU, iRNN = nn.iRNN}
    local rnn = rnnTypes[opt.model]
    local hiddenSize = opt.embeddingSize
    modelConfig.recurrentEncoder = nn.Sequential()
    for i=1, opt.numLayers do
      modelConfig.recurrentEncoder:add(rnn(hiddenSize, opt.rnnSize, opt.initWeight))
      if opt.dropout > 0 then
        modelConfig.recurrentEncoder:add(nn.Dropout(opt.dropout))
      end
      hiddenSize = opt.rnnSize
    end

    hiddenSize = opt.embeddingSize
    modelConfig.recurrentDecoder = nn.Sequential()
    for i=1, opt.numLayers do
      modelConfig.recurrentDecoder:add(rnn(hiddenSize, opt.rnnSize, opt.initWeight))
      if opt.dropout > 0 then
        modelConfig.recurrentEncoder:add(nn.Dropout(opt.dropout))
      end
      hiddenSize = opt.rnnSize
    end
    modelConfig.recurrentDecoder = modelConfig.recurrentEncoder:clone()
    modelConfig.embedder = nn.LookupTable(vocabSize, opt.embeddingSize)
    modelConfig.classifier = nn.Linear(opt.rnnSize, vocabSize)
    --  modelConfig.classifier = nn.TemporalConvolution(opt.rnnSize, vocabSize, 1)
end

modelConfig.embedder:cuda()
local enc = nn.Sequential():add(modelConfig.embedder):add(nn.Reverse(2):cuda()):add(modelConfig.recurrentEncoder)
local dec = nn.Sequential():add(modelConfig.embedder:clone('weight','gradWeight'))
dec:add(modelConfig.recurrentDecoder):add(nn.TemporalModule(modelConfig.classifier))
local trainingConfig = require 'utils.trainEncDec'
print(enc)

local logFilename = paths.concat(opt.save,'LossRate.log')
local log = optim.Logger(logFilename)
local decreaseLR = EarlyStop(1,opt.epochDecay)
local stopTraining = EarlyStop(opt.earlyStop, opt.epoch)
local epoch = 1
local sampledDec = nn.Sequential():add(modelConfig.embedder):add(modelConfig.recurrentDecoder):add(modelConfig.classifier)
repeat
    print('\nEpoch ' .. epoch ..'\n')
    --print('\nSampled Text:\n' .. sample('once again the specialists were not able to handle the imbalances on the floor of the new york stock exchange'))
    --print('\nSampled Text:\n' .. sample('a form of asbestos once used to make kent cigarette filters has caused a high percentage of cancer deaths among a group of workers'))
    --print('\nSampled Text:\n' .. sample("the following were among yesterday 's offerings and pricings in the u.s. and non-u.s. capital markets"))
    --print('\nSampled Text:\n' .. sample('closely held central diagnostic laboratory inc. in a cash and securities transaction valued at $ N million'))
    print('\nSampled Text:\n' .. sample({enc, data.encodeFunc}, {sampledDec, data.decodeFunc},'two weeks ago viewers started calling a  number for advice on various issues'))
    --LossTrain = train(data.trainingData)
    LossTrain = seq2seq({data.sentences, enc, data.vocab},{data.sentences, dec, data.vocab})
    saveModel(epoch)
    if opt.optState then
        torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
    end
    print('\nTraining Perplexity: ' .. torch.exp(LossTrain))

    local LossVal = evaluate(data.validationData)

    print('\nValidation Perplexity: ' .. torch.exp(LossVal))

    local LossTest = evaluate(data.testData)


    print('\nTest Perplexity: ' .. torch.exp(LossTest))
    log:add{['Training Loss']= LossTrain, ['Validation Loss'] = LossVal, ['Test Loss'] = LossTest}
    log:style{['Training Loss'] = '-', ['Validation Loss'] = '-', ['Test Loss'] = '-'}
    log:plot()
    epoch = epoch + 1

    if decreaseLR:update(LossVal) then
        optimState.learningRate = optimState.learningRate / opt.decayRate
        print("Learning Rate decreased to: " .. optimState.learningRate)
        decreaseLR = EarlyStop(1,1)
        decreaseLR:reset()
    end

until stopTraining:update(LossVal)

local lowestLoss, bestIteration = stopTraining:lowest()

print("Best Iteration was " .. bestIteration .. ", With a validation loss of: " .. lowestLoss)
