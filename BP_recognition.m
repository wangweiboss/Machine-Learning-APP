%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BP_recognition.m
% ʹ��ANN��������ʶ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc
% trainset=1:10;%ѵ����Ϊÿ�˵�ǰ8��
% testset=1:10;%ѵ����Ϊû�˵ĺ�2��
for u=1:10
trainset=[1:u-1,u+1:10];
testset=u;

class=40;%����40�ˡ�����40��
size1=112*92;%ÿ��ͼ�Ĵ�С
path='F:\matlab\����ʶ�����հ�\ORL\s';

%% �ع�ѵ����training set
% step1.��set����PCA��ά
% step2.���ǩ
[P,T,base]=rebuild_traindataset(path,class,trainset,size1);

%% �ع����Լ�test set
[P_test,T_test]=rebuild_testdataset(path,class,testset,size1,base);

%% ���ع���ѵ����ѵ��BP����
net=newcf(minmax(P),T,[class*length(trainset),class],{'tansig' 'logsig'},'trainscg');
net.trainparam.epochs=5000;
net.trainparam.goal=0.0003;
net.divideFcn = '';
net=train(net,P,T);
save bpnet8 net
%% ����
%ֻʹ��һ�������������в���
a=sim(net,P_test(:,2));
fprintf('��ͼƬ�ǣ�s%d\n',find(a==max(a)))
fprintf('ʵ�ʸ�ͼƬ�ǣ�s%d\n',find(T_test(:,2)==0.9))
%% ͳ����ȷ��
acc=0;
b=sim(net,P_test);
for i=1:class*length(testset)
    if find(T_test(:,i)==0.9)==find(b(:,i)==max(b(:,i)))
        acc=acc+1;
    else
%         fprintf('%d��ͼƬʵ���ǣ�s%d�����ֳ��ˣ�s%d\n',i,find(T_test(:,i)==0.9),find(b(:,i)==max(b(:,i))))
    end
end
fprintf('%d��ѵ��ͼƬ,%d�Ų���ͼƬ��׼ȷ��Ϊ��%d\n',class*length(trainset),class*length(testset),acc/(class*length(testset)));
end