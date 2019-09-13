fileFolder=fullfile('C:\Users\yy\Desktop\data\VOCdevkit2007_new\JPEGImages\');
 caffe.reset_all();
 rcnn_model = fast_rcnn_load_net();%load model
dirOutput=dir(fullfile(fileFolder,'*.xml'));
fileNames={dirOutput.name}';
length =  size(fileNames,1);
all_all_feat = [];
for i=1:length
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
        if label_name == 83
            label_name = 0;
        else if label_name == 58
                label_name = 1;
            else if label_name == 36
                    label_name = 2;
                else if label_name == 27
                        label_name = 3;
                    else if label_name == 18
                            label_name = 4;
                        else if label_name == 12
                                label_name = 5;
                            else if label_name == 9
                                    label_name = 6;
                                end
                            end
                        end
                    end
                end
            end
        end
                                        
        bndbox = xml_info.object(i).bndbox;
        boxes(i, :) = [label_name, bndbox.xmin, bndbox.ymin, bndbox.xmax, bndbox.ymax];
%         figure(1); showboxes(curFrame_picture, boxes(i,2:end));
    end
    rcnnfeat = fast_rcnn_im_detect(curFrame_picture, boxes(:,2:end), rcnn_model);%£¨500(Ð¡ ´ó)£©
    all_all_feat = [all_all_feat; boxes(:,1), rcnnfeat ];
end
