function net = Mu_init_net()
caffe.set_mode_gpu();       
caffe.reset_all();
deploy = 'E:\MU\LSTI_TIP\MU\Mutest.prototxt';    %3
% deploy = '.\caffe\model\LSTI_test_nosift.prototxt';   
%  caffe_model = 'D:\caffe\caffe-master\examples\mnist\nosiftflow\_iter_9000.caffemodel';    %4
caffe_model = 'D:\caffe\caffe-master\examples\mnist\_iter_11000.caffemodel';
% caffe_model = 'E:\LSTI_AAAI2018\caffe\model\LSTI.caffemodel';
net = caffe.Net(deploy, caffe_model, 'test');    