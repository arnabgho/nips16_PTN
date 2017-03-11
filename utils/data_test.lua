-- code adapted from https://github.com/soumith/dcgan.torch.git
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}

local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(n, dataset_name, opt_)
  opt_ = opt_ or {}
  local self = {}
  for k,v in pairs(data) do
    self[k] = v
  end

  local donkey_file 
  if dataset_name == 'dataset_rotatorRNN_base' then
    donkey_file = 'testset_rotatorRNN_base.lua'
  elseif dataset_name == 'dataset_rotatorRNN_curriculum' then
    donkey_file = 'testset_rotatorRNN_curriculum.lua'
  elseif dataset_name == 'dataset_ptn' then
    donkey_file = 'testset_ptn.lua'
  end

  if n > 0 then
    local options = opt_
    self.threads = Threads(n,
                            function() require 'torch' end,
                            function(idx)
                              opt = options
                              tid = idx
                              local seed = (opt.manualSeed and opt.manualSeed or 0) + idx
                              torch.manualSeed(seed)
                              torch.setnumthreads(1)
                              print(string.format('Starting donkey with id: %d seed %d', tid, seed))
                              assert(options, 'options not found')
                              assert(opt, 'opts not given')
                              paths.dofile(donkey_file)
                            end
    )
  else
    if donkey_file then paths.dofile(donkey_file) end
    self.threads = {}
    function self.threads:addjob(f1, f2) f2(f1()) end
    function self.threads:dojob() end
    function self.threads:synchronize() end
  end

  local nSamples = 0
  self.threads:addjob(function() return dataLoader:size() end,
        function(c) nSamples = c end)
  self.threads:synchronize()
  self._size = nSamples

  for i = 1, n do
    self.threads:addjob(self._getFromThreads,
                        self._pushResult)
  end

  return self
end

function data._getFromThreads()
  assert(opt.batch_size, 'opt.batch_size not found')
  return dataLoader:sample(opt.batch_size)
end

function data._pushResult(...)
  local res = {...}
  if res == nil then
    self.threads:synchronize()
  end
  result[1] = res
end

function data:getBatch()
  self.threads:addjob(self._getFromThreads, self._pushResult)
  self.threads:dojob()
  local res = result[1]
  result[1] = nil
  if torch.type(res) == 'table' then
    return unpack(res)
  end
  return res
end

function data:size()
  return self._size
end

return data
