local tds = require 'tds'
require 'vocabulary'

local function reduceNumWords(sentences, num, replaceWith)
    for _, v in pairs(sentences) do
        for i=1, v:size(1) do
            if v[i] > num then
                v[i] = replaceWith
            end
        end
    end
    return sentences
end


local seqProvider = torch.class('seqProvider')
function seqProvider:__init(...)
    local args = dok.unpack(
    {...},
    'seqProvider',
    'Sequence feeder',
    {arg='source', type='text', help='data source filename', required = true},
    {arg='vocab', type='userdata', help='vocabulary class object'},
    {arg='tokenizer', type='function', help='tokenizing function'},
    {arg='padStart', type='boolean', help='token at start of eache sequence (optional)'},
    {arg='padEnd', type='boolean', help='token at start of eache sequence (optional)'},
    {arg='minLength', type='number', help='minimum sequence length', default = 1},
    {arg='maxLength', type='number', help='maximum sequence length', default = 50},
    {arg='offsetStart', type='number', help='start relative location in source', default = 0},
    {arg='offsetEnd', type='number', help='end relative location in source', default = 0},
    {arg='type', type='string', help='type of output tensor', default = 'torch.IntTensor'},
    {arg='cacheFolder', type='text', help='cache folder', default = './cache/'},
    {arg='preprocess', type='boolean', help='save a preprocessed file', default = true},
    {arg='limitVocab', type='number', help='size limit for vocabulary'}
    )
    self.maxLength = args.maxLength
    self.minLength = args.minLength
    self.padStart = args.padStart
    self.padEnd = args.padEnd
    self.buffer = torch.Tensor():type(args.type)
    self.preprocess = args.preprocess
    self.currentIndex = 1
    self.limitVocab = args.limitVocab
    self.vocab = args.vocab
    self.vocabSize = 0
    self.firstIndex = 1 + args.offsetStart

    assert(args.offsetStart >= 0)
    assert(args.offsetEnd <= 0)

    sys.execute('mkdir -p ' .. args.cacheFolder)

    local filename = paths.basename(args.source)
    local cacheFilename = args.cacheFolder .. filename .. '_cached.t7'
    local vocabFilename = args.cacheFolder .. filename .. '_vocab.t7'

    if not self.vocab then
      if paths.filep(vocabFilename) then
        self.vocab = torch.load(vocabFilename)
      else
        self.vocab = vocabulary{source = args.source, tokenizer = args.tokenizer}
        torch.save(vocabFilename, self.vocab)
      end
    end

    if self.limitVocab then
        self.vocab:reduceFrequent(self.limitVocab)
        self.limitVocab = math.min(self.limitVocab, self.vocab:size())
    end

    if self.preprocess then
        if paths.filep(cacheFilename) then
            self.data = torch.load(cacheFilename)
        else
            self.data = self:__loadTextLinesVec(args.source, self.vocab, self.encoder, self.minLength, self.maxLength)
            torch.save(cacheFilename, self.data)
        end
        self.lastIndex = #self.data + args.offsetEnd
        self.getItemFunc = function(idx) return self.data[idx] end
        if self.limitVocab then
            self.data = reduceNumWords(self.data, self.limitVocab, self.vocab:unk())
        end
    else
        self.data = io.open(args.source, 'r')
        self.lastIndex = args.offsetEnd + tonumber(sys.execute('wc -l ' .. args.source .. " | cut -d ' ' -f 1"))
        self:reset()
        self.getItemFunc = function(idx)
            local line = self.data:read()
            return self:encode(line)
        end
    end
end

function seqProvider:encode(str)
    return self.vocab:encode(str, self.minLength, self.maxLength)
end

function seqProvider:decode(vector)
    return self.vocab:decode(vector)
end

function seqProvider:type(t)
    self.buffer = self.buffer:type(t)
    return self
end

function seqProvider:size()
  return self.lastIndex - self.firstIndex + 1
end
function seqProvider:__loadTextLinesVec(filename)
    local tds = require 'tds'
    local file = io.open(filename, 'r')
    local vector = tds.Vec()

    for line in file:lines() do
        local wordsVec = self:encode(line)
        if wordsVec then
            vector:insert(wordsVec)
        end
    end
    return vector
end


function seqProvider:getIndexes(indexes)
  local numSeqs
  local byOrder = false
  if torch.type(indexes) == 'number' then
    numSeqs = indexes
    byOrder = true
  else
    numSeqs = indexes:size(1)
  end
  local startSeq = 1
  local addedLength = 0
  local currMaxLength = 0

  if self.padStart then
    startSeq = 2
    addedLength = 1
  end
  if self.padEnd then
    addedLength = addedLength + 1
  end
  local bufferLength = self.maxLength + addedLength
  self.buffer:resize(numSeqs, bufferLength):fill(self.vocab:pad())
  if self.padStart then
    self.buffer:select(2,1):fill(self.vocab:go())
  end

  for i = 1, numSeqs do
    local idx = (not byOrder and indexes[i]) or self.currentIndex + i - 1
    local currSeq = self.getItemFunc(idx)
    if currSeq then
      local currLength = math.min(currSeq:nElement(), self.maxLength)
      currSeq = currSeq:narrow(1,1, currLength)
      currMaxLength = math.max(currMaxLength, currLength)
      self.buffer[i]:narrow(1, startSeq, currLength):copy(currSeq)
      if self.padEnd then
        self.buffer[i][currLength + addedLength] = self.vocab:eos()
      end
    end
  end
  return self.buffer:narrow(2, 1, currMaxLength + addedLength)
end

function seqProvider:getBatch(size)
    if self.currentIndex >= self.lastIndex then
        return nil
    end
    local size = math.min(size, self.lastIndex - self.currentIndex + 1)
    local batch = self:getIndexes(size)
    self.currentIndex = self.currentIndex + size
    return batch
end

function seqProvider:reset()
    self.currentIndex = self.firstIndex
    if not self.preprocess then
        self.data:seek("set")
        for i=1, self.firstIndex-1 do
          self.data:read('*l')
        end
    end
end

function seqProvider:getVocab()
  return vocab
end
