%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Knn_recognition.m
% ʹ��KNN��������ʶ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
for u=1:10
trainset=[1:u-1,u+1:10];
testset=u;


class=40;%����40�ˡ�����40��
size1=112*92;%ÿ��ͼ�Ĵ�С
path='F:\matlab\����ʶ�����հ�\ORL\s';
k=3;%ǰ3�������

%% �ع�ѵ����training set
% step1.��set����PCA��ά
% step2.���ǩ
[P,T,base]=rebuild_traindataset(path,class,trainset,size1);

%% �ع����Լ�test set
% step1.��set����PCA��ά
% step2.���ǩ
[P_test,T_test]=rebuild_testdataset(path,class,testset,size1,base);

%% ����
%ѡ��3������ڣ�ʹ�þ��뺯����'cityblock'
%ֻʹ��һ�������������в���
% index=knnsearch(P',P_test(:,1)','dist','cityblock','k',k);
% if (find(T(:,index(1))==0.9)~=find(T(:,index(2))==0.9)) & (find(T(:,index(1))==0.9)~=find(T(:,index(3))==0.9)) & (find(T(:,index(2))==0.9)~=find(T(:,index(3))==0.9))
%     fprintf('��ͼƬ�ǣ�s%d\n',find(T(:,index(1))==0.9))
% elseif find(T(:,index(1))==0.9)==find(T(:,index(2))==0.9)
%     fprintf('��ͼƬ�ǣ�s%d\n',find(T(:,index(1))==0.9))
% elseif find(T(:,index(1))==0.9)==find(T(:,index(3))==0.9)
%     fprintf('��ͼƬ�ǣ�s%d\n',find(T(:,index(1))==0.9))
% elseif find(T(:,index(2))==0.9)==find(T(:,index(3))==0.9)
%     fprintf('��ͼƬ�ǣ�s%d\n',find(T(:,index(2))==0.9))
% end
% fprintf('ʵ�ʸ�ͼƬ�ǣ�s%d\n',find(T_test(:,1)==0.9))

%% ͳ����ȷ��
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
%         fprintf('��ͼƬʵ���ǣ�s%d�����ֳ��ˣ�s%d\n',find(T_test(:,i)==0.9),t)
    end
end

fprintf('%d��ѵ��ͼƬ,%d�Ų���ͼƬ��׼ȷ��Ϊ��%d\n',class*length(trainset),class*length(testset),acc/(class*length(testset)))
end