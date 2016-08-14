%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BP_recognition.m
% 使用ANN进行人脸识别

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc
trainset=1:10;
testset=1:10;
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
T1=[];
for i=1:length(trainset)*class
    T1(i)=find(T(:,i)~=0);
end

%% 重构测试集test set
[P_test,T_test]=rebuild_testdataset(path,class,testset,size1,base);
T1_test=[];
for i=1:length(testset)*class
    T1_test(i)=find(T_test(:,i)~=0);
end

%% 训练SVM分类器
model = libsvmtrain(T1',P','-s 1 -t 2');
%matlab自带的svm叫svmtrain，为了区别，重命名libsvm的函数（前面加lib）
 % Usage: model = libsvmtrain(weight_vector, training_label_vector, training_instance_matrix, 'libsvm_options');
    % libsvm_options:
    % -s svm_type : set type of SVM (default 0)
    % 	0 -- C-SVC
    % 	1 -- nu-SVC
    % 	2 -- one-class SVM
    % 	3 -- epsilon-SVR
    % 	4 -- nu-SVR
    % -t kernel_type : set type of kernel function (default 2)
    % 	0 -- 线性：linear: u'*v
    % 	1 -- 多项式：polynomial: (gamma*u'*v + coef0)^degree
    % 	2 -- 径向基函数：radial basis function: exp(-gamma*|u-v|^2)
    % 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
    % 	4 -- 用户自定义核函数：precomputed kernel (kernel values in training_instance_matrix)
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

%% 测试
% [predictlabel,accuracy] = libsvmpredict(testdatalabel,testdata,model);
for i=1:2
    a=libsvmpredict(T1_test(i),P_test(:,i)',model)
end

%% 统计正确率
right=0;
for i=1:length(testset)*class
    a(i) = libsvmpredict(T1_test(i),P_test(:,i)',model);
    if a(i)==T1_test(i)
        right=right+1;
    end
end
acc(u)=right/(length(testset)*class);
end




