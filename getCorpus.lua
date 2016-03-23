local savedDir = arg[1] or paths.concat(paths.home, "Datasets", "Language")
local dataName = arg[2] or "books"--"news_commentary_v10"


local dataConfig = {
  news_commentary_v10 = {
    url = "http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz",
    filename = "training-parallel-nc-v10.tgz",
    cmd = "tar -xvf <FILE> -C <DIR>"
  },
  common_crawl = {
    url = "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
    filename = "training-parallel-nc-v10.tgz",
    cmd = "tar -xvf <FILE> -C <DIR>"
  },
  books = {
    {
      url = "http://www.cs.toronto.edu/~mbweb/books_large_p1.txt",
      filename = "books_large_p1.txt"
    },
    {
      url = "http://www.cs.toronto.edu/~mbweb/books_large_p2.txt",
      filename = "books_large_p2.txt"
    }
  }
}


local destFolder = paths.concat(savedDir, dataName)
os.execute("mkdir -p " .. destFolder)

if #dataConfig[dataName] < 1 then --convert to single table
  dataConfig[dataName] = {dataConfig[dataName]}
end

for _,config in pairs(dataConfig[dataName]) do
  local destFile = paths.concat(destFolder, config.filename)
  os.execute("wget -nc " .. config.url .. " -O " .. destFile)
  if config.cmd then
    local cmd = config.cmd:gsub("<FILE>", destFile):gsub("<DIR>", destFolder)
    os.execute(cmd)
  end
end
