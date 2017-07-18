require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'ptn'
require 'nngraph'
require 'optim'
require 'image'
matio=require 'matio'
--require 'mattorch'

model_utils = require 'utils.model_utils'
optim_utils = require 'utils.adam_v2'

opt= lapp[[
  --save_every        (default 20)
  --print_every       (default 1)
  --data_root         (default 'data')
  --data_id_path      (default 'data/shapenetcore_ids')
  --data_view_path    (default 'data/shapenetcore_viewdata')
  --data_vox_path     (default 'data/shapenetcore_voxdata')
  --dataset           (default 'dataset_ptn')
  --gpu               (default 1)
  --use_cudnn         (default 1)
  --nz                (default 512)
  --na                (default 3)
  --nview             (default 24)
  --nThreads          (default 4)
  --niter             (default 100)
  --display           (default 1)
  --checkpoint_dir    (default 'models/')
  --lambda_msk        (default 0)
  --lambda_vox        (default 1)
  --lambda_viewpoint  (default 0)	
  --kstep             (default 24)
  --batch_size        (default 6)
  --adam              (default 1)
  --arch_name         (default 'arch_E2E_PTN')
  --weight_decay      (default 0.001)
  --exp_list          (default 'singleclass')
  --load_size         (default 64)
  --vox_size          (default 32)
  --ncam              (default 16)
  --encoder_type      (default 'pretrained')
]]

opt.focal_length = math.sqrt(3)/2
opt.ntrain = math.huge
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if opt.gpu > 0 then
  ok, cunn = pcall(require, 'cunn')
  ok2, cutorch = pcall(require, 'cutorch')
  cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local TrainLoader = require('utils/data.lua')
local ValLoader = require('utils/data_val.lua')

local data = TrainLoader.new(opt.nThreads, opt.dataset, opt)
local data_val = ValLoader.new(opt.nThreads, opt.dataset, opt)

print("Dataset: " .. opt.dataset, "train_size: ", data:size(), "val_size: ", data_val:size())

local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') and name:find('Spatial') then
    local nin = m.nInputPlane*m.kH*m.kW
    m.weight:uniform(-0.08, 0.08):mul(math.sqrt(1/nin))
    m.bias:fill(0)
  elseif name:find('Convolution') and name:find('Volumeric') then
    local nin = m.nInputPlane*m.kT*m.kH*m.kW
    m.weight:uniform(-0.08, 0.08):mul(math.sqrt(1/nin))
    m.bias:fill(0)
  elseif name:find('Linear') then
    local nin = m.weight:size(2)
    m.weight:uniform(-0.08, 0.08):mul(math.sqrt(1/nin))
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

opt.model_name = string.format('%s_%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg(%g,%g)_ks%d_vs%d', 
  opt.encoder_type,opt.arch_name, opt.exp_list, opt.nview, opt.adam, opt.batch_size, opt.nz,
  opt.weight_decay, opt.lambda_msk, opt.lambda_vox, opt.kstep, opt.vox_size)

-- initialize parameters
init_models = dofile('scripts/' .. opt.arch_name .. '.lua')
model = {}
local projector=nil
model.encoder, model.voxel_dec, projector = init_models.create(opt)
if opt.encoder_type=='pretrained' then
    base_loader=torch.load(opt.checkpoint_dir .. 'cnn_vol.t7')
    model.encoder=base_loader.encoder
else
    model.encoder:apply(weights_init)
end

model.voxel_dec:apply(weights_init)
projector:apply(weights_init)

print(model.encoder)
opt.model_path = opt.checkpoint_dir .. opt.model_name
if not paths.dirp(opt.model_path) then
  paths.mkdir(opt.model_path)
end

-- load encoder from RNN-16
if opt.exp_list == 'singleclass' then
  opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
    'rotatorRNN1_64', opt.exp_list, opt.nview, 2, 8, opt.nz, 
    opt.weight_decay, 10, 16)
  opt.basemodel_epoch = 20 
--[[elseif opt.exp_list == 'multiclass' then
  opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
    'rotatorRNN1_64', opt.exp_list, opt.nview, 2, 8, opt.nz, 
    opt.weight_decay, 10, 16) 
  opt.basemodel_epoch = 20
  loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch))
  encoder = loader.encoder]]
end

collectgarbage()

-- load model from previos iterations
prev_iter = 0
for i = opt.niter, 1, -opt.save_every do
  print(opt.model_path .. string.format('/net-epoch-%d.t7', i))
  if paths.filep(opt.model_path .. string.format('/net-epoch-%d.t7', i)) then
    prev_iter = i
    loader = torch.load(opt.model_path .. string.format('/net-epoch-%d.t7', i))
    state = torch.load(opt.model_path .. '/state.t7')
    print(string.format('resuming from epoch %d', i))
    break
  end
end

-- build nngraph
if prev_iter > 0 then
  model.encoder = loader.encoder
  model.voxel_dec = loader.voxel_dec
  projector = loader.projector
end

-- criterion
local criterion_vox = nn.MSECriterion()
--criterion_vox.sizeAverage = false
local criterion_msk = nn.MSECriterion()
--criterion_msk.sizeAverage = false
local criterion_viewpoints=nn.MSECriterion() 

-- hyperparams
function getAdamParams(opt)
  config = {}
  if opt.adam == 1 then
    config.learningRate = 0.0001
    config.epsilon = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weightDecay = opt.weight_decay
  elseif opt.adam == 2 then
    config.learningRate = 0.1
    config.epsilon = 1e-8
    config.beta1 = 0.5
    config.beta2 = 0.999
    config.weightDecay = opt.weight_decay
  end
  return config
end

config = getAdamParams(opt)
print(config)
-------------------------------------------

local batch_im_in = torch.Tensor(opt.batch_size*opt.kstep, 3, opt.load_size, opt.load_size)
local batch_feat = torch.Tensor(opt.batch_size * opt.kstep, opt.nz)
local batch_viewpoints = torch.Tensor(opt.batch_size * opt.kstep, opt.ncam)
local batch_vox = torch.Tensor(opt.batch_size * opt.kstep, 1, opt.vox_size, opt.vox_size, opt.vox_size)
local batch_proj = torch.Tensor(opt.batch_size * opt.kstep, 1, opt.vox_size, opt.vox_size)
local batch_trans = torch.Tensor(opt.batch_size * opt.kstep, 4, 4)

local tmp_gt_im = torch.Tensor(opt.batch_size, 3, opt.load_size, opt.load_size)
local tmp_pred_proj = torch.Tensor(opt.batch_size, 1, opt.vox_size, opt.vox_size)
local tmp_gt_proj = torch.Tensor(opt.batch_size, 1, opt.vox_size, opt.vox_size)
local tmp_pred_vox = torch.Tensor(opt.batch_size,1,opt.vox_size,opt.vox_size,opt.vox_size)
local tmp_gt_vox = torch.Tensor(opt.batch_size,1,opt.vox_size,opt.vox_size,opt.vox_size)

local errVOX, errMSK, errViewpoint
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
--------------------------------------------

if opt.gpu > 0 then
  batch_im_in = batch_im_in:cuda()
  batch_feat = batch_feat:cuda()
  batch_viewpoints = batch_viewpoints:cuda()
  batch_vox = batch_vox:cuda()
  batch_proj = batch_proj:cuda()
  batch_trans = batch_trans:cuda()
  model.encoder:cuda()
  model.voxel_dec:cuda()
  projector:cuda()
  criterion_vox:cuda()
  criterion_msk:cuda()
  criterion_viewpoints:cuda()
end

--params, grads = voxel_dec:getParameters()
--paramEnc, gradEnc = encoder:getParameters()
--paramProj, gradProj = projector:getParameters()

params,grads=model_utils.combine_all_parameters(model)

-- perspective projection
--------------------------------------------------
local specify_pers_transformation = function(phi, theta, focal_length)
  local T = torch.Tensor(4, 4):zero()
  local K = torch.Tensor(4, 4):eye(4)
  local E = torch.Tensor(4, 4):eye(4)

  local sin_phi = math.sin(phi*math.pi/180.0)
  local cos_phi = math.cos(phi*math.pi/180.0)
  local sin_theta = math.sin((-theta)*math.pi/180.0)
  local cos_theta = math.cos((-theta)*math.pi/180.0)
  -- rotation axis -- z
  R_azi = torch.Tensor(3, 3):zero()
  R_azi[1][1] = cos_theta
  R_azi[3][3] = cos_theta
  R_azi[1][3] = -sin_theta
  R_azi[3][1] = sin_theta
  R_azi[2][2] = 1

  -- rotation axis -- x
  R_ele = torch.Tensor(3, 3):zero()
  R_ele[1][1] = cos_phi
  R_ele[1][2] = sin_phi
  R_ele[2][1] = -sin_phi
  R_ele[2][2] = cos_phi
  R_ele[3][3] = 1
  R_comb = R_azi * R_ele

  local colR = torch.Tensor(3,1):zero()
  --local focal_length = math.sqrt(3)/2
  colR[1][1] = opt.focal_length + math.sqrt(1)/2
  colR = R_comb * colR
  E[{{1,3}, {1,3}}] = R_comb:clone()
  E[{{1,3}, {4}}] = -colR:clone()
  
  K[3][3] = 1/opt.focal_length
  K[2][2] = 1/opt.focal_length
  T = E * K

  return T
end

local getTransMatrix = function(vid)
  local T = specify_pers_transformation(30, vid*15, opt.focal_length)
  return T
end
--------------------------------------------------
local opfunc = function(x)
  collectgarbage()
  if x ~= params then
    params:copy(x)
  end
  grads:zero()
  -- train
  data_tm:reset(); data_tm:resume()
  cur_train_ims, cur_train_vox, _ = data:getBatch()
  data_tm:stop()

  for m = 1, opt.batch_size do
    for k = 1, opt.kstep do
	  batch_im_in[(m-1)*opt.kstep+k]:copy(cur_train_ims[m][k]:mul(2):add(-1))
      batch_vox[(m-1)*opt.kstep+k]:copy(cur_train_vox[m])
      batch_trans[(m-1)*opt.kstep+k]:copy(getTransMatrix(k))
    end
  end

  --local f_id = model.encoder:forward(batch_im_in)[1]:clone()
  --for m = 1, opt.batch_size do
  --  for k = 1, opt.kstep do
  --    batch_feat[(m-1)*opt.kstep+k]:copy(f_id[m])
  --  end
  --end

  local d_encoder_viewpoints=nil
  local errViewpoint=0
  if opt.encoder_type=='viewpoint_regress' then
      local encoded=model.encoder:forward(batch_im_in) 
      batch_feat=encoded[1]
      batch_viewpoints=encoded[2]
      errViewpoint= criterion_viewpoints:forward(batch_viewpoints,batch_trans:reshape(opt.batch_size*opt.kstep,opt.ncam))
      d_encoder_viewpoints= criterion_viewpoints:backward(batch_viewpoints,batch_trans:reshape(opt.batch_size*opt.kstep,opt.ncam)):mul(opt.lambda_viewpoint) 
  elseif opt.encoder_type=='viewpoint_input' then
      batch_feat=model.encoder:forward({batch_im_in,batch_trans:reshape(opt.batch_size*opt.kstep,opt.ncam,1,1)} )
  elseif opt.encoder_type=='viewpoint_oblivious' then
      batch_feat=model.encoder:forward(batch_im_in)
  elseif opt.encoder_type=='pretrained' then
      batch_feat=model.encoder:forward(batch_im_in)[1]
  end
  batch_proj = projector:forward({batch_vox, batch_trans}):clone()

  local f_vox = model.voxel_dec:forward(batch_feat)
  local f_proj = projector:forward({f_vox, batch_trans})
 
  errVOX = criterion_vox:forward(f_vox, batch_vox) 
  local df_dVOX = criterion_vox:backward(f_vox, batch_vox):mul(opt.lambda_vox)
  
  errMSK = criterion_msk:forward(f_proj, batch_proj) 
  local df_dMSK = criterion_msk:backward(f_proj, batch_proj):mul(opt.lambda_msk)

  local df_dproj = projector:backward({f_vox, batch_trans}, df_dMSK)
  local df_dvox = model.voxel_dec:backward(batch_feat, df_dproj[1]:clone() + df_dVOX:clone())
  
  if opt.encoder_type=='viewpoint_regress' then
    local df_d_im_in=model.encoder:backward(batch_im_in,{df_dvox,d_encoder_viewpoints}) 
  elseif opt.encoder_type=='viewpoint_input' then
    local df_d_im_in=model.encoder:backward({batch_im_in,batch_trans:reshape(opt.batch_size*opt.kstep,opt.ncam,1,1)},df_dvox) 
  elseif opt.encoder_type=='viewpoint_oblivious' then
    local df_d_im_in=model.encoder:backward(batch_im_in,df_dvox) 
  end
  local err = errVOX * opt.lambda_vox + errMSK * opt.lambda_msk + errViewpoint * opt.lambda_viewpoint

  return err, grads
end
--------------------------------------------------------
-- New Feedforward Function
--------------------------------------------------------

local feedforward = function(x)
  collectgarbage()
  if x ~= params then
    params:copy(x)
  end
  grads:zero()
  -- train
  data_tm:reset(); data_tm:resume()
  cur_ims, cur_vox, _ = data_val:getBatch()
  data_tm:stop()

  for m = 1, opt.batch_size do
    for k = 1, opt.kstep do
	  batch_im_in[(m-1)*opt.kstep+k]:copy(cur_ims[m][k]:mul(2):add(-1))
      batch_vox[(m-1)*opt.kstep+k]:copy(cur_vox[m])
      batch_trans[(m-1)*opt.kstep+k]:copy(getTransMatrix(k))
    end
  end

  --local f_id = model.encoder:forward(batch_im_in)[1]:clone()
  --for m = 1, opt.batch_size do
  --  for k = 1, opt.kstep do
  --    batch_feat[(m-1)*opt.kstep+k]:copy(f_id[m])
  --  end
  --end


  local errViewpoint=0
  if opt.encoder_type=='viewpoint_regress' then
      local encoded=model.encoder:forward(batch_im_in) 
      batch_feat=encoded[1]
      batch_viewpoints=encoded[2]
      errViewpoint= criterion_viewpoints:forward(batch_viewpoints,batch_trans:reshape(opt.batch_size*opt.kstep,opt.ncam))
  elseif opt.encoder_type=='viewpoint_input' then
      batch_feat=model.encoder:forward({batch_im_in,batch_trans:reshape(opt.batch_size*opt.kstep,opt.ncam,1,1)} )
  elseif opt.encoder_type=='viewpoint_oblivious' then
      batch_feat=model.encoder:forward(batch_im_in)
  elseif opt.encoder_type=='pretrained' then
      batch_feat=model.encoder:forward(batch_im_in)[1]
  end


  batch_proj = projector:forward({batch_vox, batch_trans}):clone()

  local f_vox = model.voxel_dec:forward(batch_feat)
  local f_proj = projector:forward({f_vox, batch_trans})
 
  errVOX = criterion_vox:forward(f_vox, batch_vox) 
  
  errMSK = criterion_msk:forward(f_proj, batch_proj) 
  for m = 1, opt.batch_size do
    k = torch.random(opt.kstep)
    tmp_gt_im[m] = batch_im_in[(m-1)*opt.kstep+k]:float():clone()
    tmp_pred_proj[m] = f_proj[(m-1)*opt.kstep+k]:float():clone()
    tmp_pred_vox[m] = f_vox[(m-1)*opt.kstep+k]:float():clone()
    tmp_gt_vox[m]= cur_vox[m]:float():clone()
    tmp_gt_proj[m] = batch_proj[(m-1)*opt.kstep+k]:float():clone()
  end

  local err = errVOX * opt.lambda_vox + errMSK * opt.lambda_msk + errViewpoint * opt.lambda_viewpoint

  return err
end

-- train & val
for epoch = prev_iter + 1, opt.niter do
  epoch_tm:reset()
  local counter = 0
  -- train
  model.encoder:training()
  model.voxel_dec:training()
  projector:training()

  for i = 1, math.max(1,math.min(data:size() / (opt.batch_size), opt.ntrain)) do
    tm:reset()
    optim_utils.adam_v2(opfunc, params, config, state)
    counter = counter + 1
    print(string.format('Epoch: [%d][%8d / %8d]\t Time: %.3f DataTime: %.3f  '
      .. ' Err_Vox: %.4f, Err_Msk: %.4f', epoch, i-1,
      math.min(data:size() / (opt.batch_size), opt.ntrain),
      tm:time().real, data_tm:time().real,
      errVOX and errVOX or -1, errMSK and errMSK or -1))
  end

  -- val
  model.encoder:evaluate()
  model.voxel_dec:evaluate()
  projector:evaluate()

  --for i = 1, 1 do
  tm:reset()
  local to_plot = {}

  for i = 1, 24 / opt.batch_size do
    local err = feedforward(params)

    for j = 1, opt.batch_size do
      local res = tmp_gt_im[j]:float():clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()

      local res = tmp_pred_proj[j]:float():clone()
      res = torch.squeeze(res)
      res = res:repeatTensor(3, 1, 1)
      res = image.vflip(res)
      res = image.scale(res, opt.load_size, opt.load_size)
      res:mul(-1):add(1)
      to_plot[#to_plot+1] = res:clone()

      local res = tmp_gt_proj[j]:float():clone()
      res = torch.squeeze(res)
      res = res:repeatTensor(3, 1, 1)
      res = image.vflip(res)
      res = image.scale(res, opt.load_size, opt.load_size)
      res:mul(-1):add(1)
      to_plot[#to_plot+1] = res:clone()
    end
  end

  local formatted = image.toDisplayTensor({input=to_plot, nrow = 12})
  formatted = formatted:double()
  formatted:mul(255)

  formatted = formatted:byte()
  image.save(opt.model_path .. string.format('/sample-%03d.jpg', epoch), formatted)
  
  local vox_dir=opt.model_path .. string.format('/vox_%03d',epoch)
  paths.mkdir(vox_dir)
  for i=1,opt.batch_size do
      matio.save(vox_dir .. string.format('/gt_%03d.mat',i),tmp_gt_vox[i])
      matio.save(vox_dir ..  string.format('/pred_%03d.mat',i),tmp_pred_vox[i])
  end

  if epoch % opt.save_every == 0 then
    torch.save((opt.model_path .. string.format('/net-epoch-%d.t7', epoch)), 
      {encoder = model.encoder, voxel_dec = model.voxel_dec, projector = projector})
    torch.save((opt.model_path .. '/state.t7'), state)
  end
end
