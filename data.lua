local tds = require 'tds'
local tokens = {'<UNK>', '<EOS>', '<GO>', '<PAD>'}



local function decodeFunc(decoder, seperator)
  local seperator = seperator or ' '
  local func = function(vec)
    local output = ''
    for i=1, vec:size(1) do
      local decoded = decoder[vec[i]] or '<UNK>'
      output = output .. seperator .. decoded
    end
    return output
  end
  return func
end

local function encodeFunc(vocab, missingVal)
  local missingVal = missingVal or -1
  local func = function(str)
    local words = str:split(' ')
    local length = #words
    local encoded = torch.IntTensor(#words):zero()

    for i=1, length do
      local currWord = words[i]
      local encodedNum = vocab[currWord]
      if not encodedNum then
        encodedNum = missingVal
      end
      encoded[i] = encodedNum
    end
    return encoded
  end

  return func
end

function loadTextLines(filename, defaultTokens, minLength, maxLength)
  local minLength = minLength or 1
  local maxLength = maxLength or 50
  local vocabSizeLimit = 2^31 - 1 --integer limit
  local removedSymbols = '[\n,"?.!/\\<>():;]'
  local vocabFreq = torch.IntTensor(vocabSizeLimit):zero()
  local tds = require 'tds'
  local file = io.open(filename, 'r')
  local vector = tds.Vec()

  local defaultTokens = defaultTokens or {}
  local vocab = {}
  local currentNum = 1
  --count num words (in case of existing vocab)
  for num,token in pairs(defaultTokens) do
    vocab[token] = num
    vocabFreq[num] = vocabSizeLimit
    currentNum = currentNum + 1
  end


  for line in file:lines() do
    local line = line:gsub(removedSymbols, ' ')
    line = line:gsub('%s+',' '):gsub('%s$',''):gsub("'s", " 's"):lower()
    local words = line:split(' ')
    local length = #words
    if length > minLength and length < maxLength then
      local wordsVec = torch.IntTensor(#words):zero()

      for i=1, length do
        local currWord = words[i]
        local encodedNum = vocab[currWord]
        if not encodedNum then
          vocab[currWord] = currentNum
          encodedNum = currentNum
          currentNum = currentNum + 1
        end
        vocabFreq[encodedNum] = vocabFreq[encodedNum] + 1
        wordsVec[i] = encodedNum
      end
      vector:insert(wordsVec)
    end
  end

  vocabFreq = vocabFreq:narrow(1, 1, currentNum - 1)
  local sortedIdxs
  vocabFreq, sortedIdxs = vocabFreq:sort(1, true)
  local newIdxs = torch.LongTensor(currentNum - 1)
  for i = 1, currentNum -1 do
    newIdxs[sortedIdxs[i]] = i
  end

  for _, v in pairs(vector) do
    for i=1, v:size(1) do
      v[i] = newIdxs[v[i]]
    end
  end

  local decodeTable = {}
  for word, num in pairs(vocab) do
    vocab[word] = newIdxs[num]
    decodeTable[newIdxs[num]] = word
  end

  local data = {
    sentences = vector,
    vocab = vocab,
    decodeTable = decodeTable,
    decodeFunc = decodeFunc(decodeTable),
    encodeFunc = encodeFunc(vocab),
    count = vocabFreq
  }
  return data
end

function reduceNumWords(sentences, num, replaceWith)
  for _, v in pairs(sentences) do
    for i=1, v:size(1) do
      if v[i] > num then
        v[i] = replaceWith
      end
    end
  end
  return sentences
end


local dataFolder = '../../Datasets/training-monolingual/'
local cacheFolder = './cache/'
sys.execute('mkdir -p ' .. cacheFolder)

local filename = 'news-commentary-v6.en'--'europarl-v6.en'
local cacheFilename = cacheFolder .. filename .. '_cached.t7'

local data
if paths.filep(cacheFilename) then
  data = torch.load(cacheFilename)
else
  data = loadTextLines(dataFolder .. filename ,tokens)
  torch.save(cacheFilename, data)
end
return data
