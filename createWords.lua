
local tds = require 'tds'
local vocab = {}
for i=1, 255 do
  vocab[string.char(i)] = i
end
vocab['<GO>'] = string.byte('\n')
vocab['<PAD>'] = string.byte(' ')
vocab['<EOS>'] = string.byte('\t')

local decodeTbl = {}
for w,n in pairs(vocab) do
  decodeTbl[n] = w
end

function decode(x)
  local s =''
  for i=1,x:size(1) do
    s = s .. decodeTbl[x[i]]
  end
  return s

end

function encode(s)
  if vocab[s] then
    x = torch.ByteTensor({vocab[s], vocab['<PAD>']})
  else
    x=torch.ByteTensor(#s)
    for i=1, x:size(1) do
      x[i] = vocab[s:sub(i,i)]
    end
  end
  return x
end



--local d = torch.load('words.t7')
local d = torch.load('ptb.t7')

local vector = tds.Vec(d)
print(#vector)
d = nil
collectgarbage()
for i=1,#vector do
  vector[i] = encode(vector[i])
end


local vocabSize = 255


local data = {
  sentences = vector,
  vocab = vocab,
  decodeTable = decodeTbl,
  decodeFunc = decode,
  encodeFunc = encode
}
torch.save('./cache/ptb_cached.t7', data)

--torch.save('./cache/wordData_cached.t7', data)
