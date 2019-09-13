function  [opts, proposal_detection_model, rpn_net]=FsterRcnnInit()
 caffe.reset_all();
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 0;
% active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;

opts.test_scales            = 600;

%% -------------------- INIT_MODEL --------------------
model_dir                   = './caffe/faster_rcnn-master/experiments/output/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers'; %% VGG-16
%model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_ZF'); %% ZF
proposal_detection_model    = load_proposal_detection_model(model_dir);

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
end

% caffe.init_log(fullfile(pwd, 'caffe_log'));
% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
% fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
% fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

% set gpu/cpu
    caffe.set_mode_gpu();
end

