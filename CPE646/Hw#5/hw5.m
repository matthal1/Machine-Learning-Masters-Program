clear



d11=[2;2]*ones(1,70)+2.*randn(2,70);
d12=[-2;-2]*ones(1,30)+randn(2,30);
d1=[d11,d12];

d21=[3;-3]*ones(1,50)+randn([2,50]);
d22=[-3;3]*ones(1,50)+randn([2,50]);
d2=[d21,d22];


hw5_1=d1;
hw5_2=d2;

save hw5.mat hw5_1 hw5_2


x1=hw5_1;
x2=hw5_2;

plot(x1(1,:),x1(2,:),'o',x2(1,:),x2(2,:),'*')
