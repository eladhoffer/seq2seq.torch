
local function decodeFunc(decoder, seperator)
  local seperator = seperator or ' '
  local func = function(vec)
    local output = ''
    for i=1, vec:size(1) do
      output = output .. space .. decoder[vec[i]]
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
    local encoded = torch.LongTensor(#words):zero()

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

function loadTextLines(filename, defaultTokens)
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
    if length > 0 then
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
    encodeFunc = encodeFunc(vocab, vocab['<UNK>']),
    count = count
  }
  return data
end

function reduceNumWords(sentences, num)
  for _, v in pairs(vector) do
    v:clamp(-1, num)
  end
return sentences
end
