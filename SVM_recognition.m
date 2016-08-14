%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BP_recognition.m
% ʹ��ANN��������ʶ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc
trainset=1:10;
testset=1:10;
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
T1=[];
for i=1:length(trainset)*class
    T1(i)=find(T(:,i)~=0);
end

%% �ع����Լ�test set
[P_test,T_test]=rebuild_testdataset(path,class,testset,size1,base);
T1_test=[];
for i=1:length(testset)*class
    T1_test(i)=find(T_test(:,i)~=0);
end

%% ѵ��SVM������
model = libsvmtrain(T1',P','-s 1 -t 2');
%matlab�Դ���svm��svmtrain��Ϊ������������libsvm�ĺ�����ǰ���lib��
 % Usage: model = libsvmtrain(weight_vector, training_label_vector, training_instance_matrix, 'libsvm_options');
    % libsvm_options:
    % -s svm_type : set type of SVM (default 0)
    % 	0 -- C-SVC
    % 	1 -- nu-SVC
    % 	2 -- one-class SVM
    % 	3 -- epsilon-SVR
    % 	4 -- nu-SVR
    % -t kernel_type : set type of kernel function (default 2)
    % 	0 -- ���ԣ�linear: u'*v
    % 	1 -- ����ʽ��polynomial: (gamma*u'*v + coef0)^degree
    % 	2 -- �����������radial basis function: exp(-gamma*|u-v|^2)
    % 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
    % 	4 -- �û��Զ���˺�����precomputed kernel (kernel values in training_instance_matrix)
    % -d degree : set degree in kernel function (default 3)
    % -g gamma : set gamma in kernel function (default 1/num_features)
    % -r coef0 : set coef0 in kernel function (default 0)
    % -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    % -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    % -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
    % -m cachesize : set cache memory size in MB (default 100)
    % -e epsilon : set tolerance of termination criterion (default 0.001)
    % -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    % -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
    % -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
    % -v n : n-fold cross validation mode
    % -q : quiet mode (no outputs)

%% ����
% [predictlabel,accuracy] = libsvmpredict(testdatalabel,testdata,model);
for i=1:2
    a=libsvmpredict(T1_test(i),P_test(:,i)',model)
end

%% ͳ����ȷ��
right=0;
for i=1:length(testset)*class
    a(i) = libsvmpredict(T1_test(i),P_test(:,i)',model);
    if a(i)==T1_test(i)
        right=right+1;
    end
end
acc(u)=right/(length(testset)*class);
end




