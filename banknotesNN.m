%% Bank note authentication

% Author-S.Badhri Narayanan

clear;
clc;

theta1=rand(6,5);
theta2=rand(1,7);
X=load('banknoteclass.txt');
x=X(:,1:4);
y=X(:,5);
m=length(x);
p=randperm(m,round(0.2*m));
p=p(:);
x=[ones(m,1),x];
xtest=x(p,:);
ytest=y(p);
x(p,:)=[];
y(p)=[];
m=length(x);
% Parameter initialization

alpha=0.5;
lambda=10^(-4);
noofiter=800;
% Gradient Descent

i=1;
while i<=noofiter
    bdel1=zeros(6,5);
    bdel2=zeros(1,7);
    
    % Forward propogation
    
    z2=theta1*x';
    a2=sigmoid(z2);
    a2=[ones(1,m);a2];
    z3=theta2*a2;
    z3=z3(:);
    h=sigmoid(z3);
    
    % Backpropogation
    
    del3=h-y;
    del2=(theta2'*del3').*(a2.*(1-a2));
    del2(1,:)=[];
    bdel1=del2*x;
    bdel2=(del3')*a2';
    
    D1=bdel1/m;
    D2=bdel2/m;
    D1(:,2:end)=D1(:,2:end)+(lambda*theta1(:,2:end));
    D2(:,2:end)=D2(:,2:end)+(lambda*theta2(:,2:end));
    
    theta1=theta1-alpha*D1;
    theta2=theta2-alpha*D2;
    i=i+1;
end

% Testing performance

mt=length(xtest);
zt2=theta1*xtest';
at2=sigmoid(zt2);
at2=[ones(1,mt);at2];
zt3=theta2*at2;
zt3=zt3(:);
ht=sigmoid(zt3);
htt=ht>=0.5;

error=abs(htt-ytest);
errors=sum(error~=0);
accuracy=(1-(errors/mt))*100

% Training performance

ztt2=theta1*x';
att2=sigmoid(ztt2);
att2=[ones(1,m);att2];
ztt3=theta2*att2;
ztt3=ztt3(:);
htt=sigmoid(ztt3);

htt=htt>=0.5;
err=abs(htt-y);
err=sum(err~=0);
train_accuracy=(1-(err/m))*100


