close all
pppp= mapminmax('reverse',P,PS);
for j=10:15:199
PPP=pppp(370,1:j)*pinv(base(:,1:j));



PPP=PPP+samplemean;

a=PPP(1:112);
for i=2:92
    a=[a;PPP((1+112*(i-1)):(112+112*(i-1)))];
end
subplot(4,4,(j-10)/15+1),imshow(a',[]),title([num2str(j) '个特征脸'])
end
b=imread(strcat(path,num2str(37),'\',num2str(1),'.pgm'));
subplot(4,4,(j-10)/15+2),imshow(b),title('原图')
