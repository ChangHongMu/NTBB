fileFolder=fullfile('C:\Users\yy\Desktop\测试跟实验数据\测试照片\');

dirOutput=dir(fullfile(fileFolder,'*.xml'));
fileNames={dirOutput.name}';
length =  size(fileNames,1);
% all_all_feat = [];
for i=1:length
    caffe.reset_all();
    rcnn_model = fast_rcnn_load_net();%load model
    xml_name = fileNames{i};
    xml_name
    xml_path = [fileFolder xml_name];
    xml_info = xml_read(xml_path);
    Path  = xml_info.path;
    curFrame_picture = imread(Path);
    LengthBoxes =  size(xml_info.object, 1);
    boxes = [];
    for i = 1:LengthBoxes
        label_name = xml_info.object(i).name;                                     
        bndbox = xml_info.object(i).bndbox;
        boxes(i, :) = [label_name, bndbox.xmin, bndbox.ymin, bndbox.xmax, bndbox.ymax];
%         figure(1); showboxes(curFrame_picture, boxes(i,2:end));
    end
    rcnnfeat = fast_rcnn_im_detect(curFrame_picture, boxes(:,2:end), rcnn_model);%（500(小 大)）
%     all_all_feat = [all_all_feat; boxes(:,1), rcnnfeat ];
    net = Mu_init_net();
    res = PredictHighVec(net, rcnnfeat);
    [maxnum ind]=max(res, [], 2);
%     aa= [24.043; 23.077;16.303; 12.620;12.280;8.923;5.553]; 
    aa = [3.991; 12.065; 31.194; 42.076; 50.097; 56.310; 59.393]; 

    index_all = [boxes(:,2:end), aa(ind)];
    
    figure(1); showboxes(curFrame_picture, index_all);
    
%     [maxnum ind]=max(res, [], 2);
% 
%     boxes_cell = cell(7, 1);
%     [m,n]=size(ind);
%     for j = 1:m
%         boxes_cell{ind(j)}=[boxes(j,2:end), 1];
%     end
%     index = cell(7,1);
%     aa= [24.043; 23.077;16.303; 12.620;12.280;8.923;5.553]; 
%     for mm = 1:7
%          index{mm} = num2str(aa(mm));
%     end
%     
%    showboxesa(curFrame_picture, boxes_cell, index, 'voc');
end
