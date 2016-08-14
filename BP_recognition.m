%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BP_recognition.m
% 使用ANN进行人脸识别

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc
% trainset=1:10;%训练集为每人的前8张
% testset=1:10;%训练集为没人的后2张
for u=1:10
trainset=[1:u-1,u+1:10];
testset=u;

class=40;%共有40人。即有40类
size1=112*92;%每张图的大小
path='F:\matlab\人脸识别最终版\ORL\s';

%% 重构训练集training set
% step1.对set进行PCA降维
% step2.打标签
[P,T,base]=rebuild_traindataset(path,class,trainset,size1);

%% 重构测试集test set
[P_test,T_test]=rebuild_testdataset(path,class,testset,size1,base);

%% 用重构的训练集训练BP网络
net=newcf(minmax(P),T,[class*length(trainset),class],{'tansig' 'logsig'},'trainscg');
net.trainparam.epochs=5000;
net.trainparam.goal=0.0003;
net.divideFcn = '';
net=train(net,P,T);
save bpnet8 net
%% 测试
%只使用一个测试样本进行测试
a=sim(net,P_test(:,2));
fprintf('该图片是：s%d\n',find(a==max(a)))
fprintf('实际该图片是：s%d\n',find(T_test(:,2)==0.9))
%% 统计正确率
acc=0;
b=sim(net,P_test);
for i=1:class*length(testset)
    if find(T_test(:,i)==0.9)==find(b(:,i)==max(b(:,i)))
        acc=acc+1;
    else
%         fprintf('%d该图片实际是：s%d，被分成了：s%d\n',i,find(T_test(:,i)==0.9),find(b(:,i)==max(b(:,i))))
    end
end
fprintf('%d张训练图片,%d张测试图片，准确率为：%d\n',class*length(trainset),class*length(testset),acc/(class*length(testset)));
end