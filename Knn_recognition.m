%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Knn_recognition.m
% 使用KNN进行人脸识别

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
for u=1:10
trainset=[1:u-1,u+1:10];
testset=u;


class=40;%共有40人。即有40类
size1=112*92;%每张图的大小
path='F:\matlab\人脸识别最终版\ORL\s';
k=3;%前3个最近邻

%% 重构训练集training set
% step1.对set进行PCA降维
% step2.打标签
[P,T,base]=rebuild_traindataset(path,class,trainset,size1);

%% 重构测试集test set
% step1.对set进行PCA降维
% step2.打标签
[P_test,T_test]=rebuild_testdataset(path,class,testset,size1,base);

%% 测试
%选用3个最近邻，使用距离函数是'cityblock'
%只使用一个测试样本进行测试
% index=knnsearch(P',P_test(:,1)','dist','cityblock','k',k);
% if (find(T(:,index(1))==0.9)~=find(T(:,index(2))==0.9)) & (find(T(:,index(1))==0.9)~=find(T(:,index(3))==0.9)) & (find(T(:,index(2))==0.9)~=find(T(:,index(3))==0.9))
%     fprintf('该图片是：s%d\n',find(T(:,index(1))==0.9))
% elseif find(T(:,index(1))==0.9)==find(T(:,index(2))==0.9)
%     fprintf('该图片是：s%d\n',find(T(:,index(1))==0.9))
% elseif find(T(:,index(1))==0.9)==find(T(:,index(3))==0.9)
%     fprintf('该图片是：s%d\n',find(T(:,index(1))==0.9))
% elseif find(T(:,index(2))==0.9)==find(T(:,index(3))==0.9)
%     fprintf('该图片是：s%d\n',find(T(:,index(2))==0.9))
% end
% fprintf('实际该图片是：s%d\n',find(T_test(:,1)==0.9))

%% 统计正确率
acc=0;
index=knnsearch(P',P_test','dist','cityblock','k',k);
for i=1:class*length(testset)
    if (find(T(:,index(i,1))==0.9)~=find(T(:,index(i,2))==0.9)) & (find(T(:,index(i,1))==0.9)~=find(T(:,index(i,3))==0.9)) & (find(T(:,index(i,2))==0.9)~=find(T(:,index(i,3))==0.9))
        t=find(T(:,index(i,1))==0.9);
    elseif find(T(:,index(i,1))==0.9)==find(T(:,index(i,2))==0.9)
        t=find(T(:,index(i,1))==0.9);
    elseif find(T(:,index(i,1))==0.9)==find(T(:,index(i,3))==0.9)
        t=find(T(:,index(i,1))==0.9);
    elseif find(T(:,index(i,2))==0.9)==find(T(:,index(i,3))==0.9)
        t=find(T(:,index(i,2))==0.9);
    end
    if t==find(T_test(:,i)==0.9)
        acc=acc+1;
    else
%         fprintf('该图片实际是：s%d，被分成了：s%d\n',find(T_test(:,i)==0.9),t)
    end
end

fprintf('%d张训练图片,%d张测试图片，准确率为：%d\n',class*length(trainset),class*length(testset),acc/(class*length(testset)))
end