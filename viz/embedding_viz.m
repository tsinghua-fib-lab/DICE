%% Movielens item embeddings (Figure 5(b) left)
data = load('./data/dice_ml.txt');
pop = load('./data/ml_popularity.txt');
mypop = zeros(4819,1);
mypop(pop<17)=1;
mypop((pop>=17) & (pop<71))=2;
mypop(pop>71)=3;

figure();
scatter(data(1:4819,1), data(1:4819,2),40,mypop,'x','DisplayName', 'DICE-int');hold on;
scatter(data(4820:9638,1), data(4820:9638,2),40,mypop+3,'.','DisplayName', 'DICE-con');hold on;
plot([-50,60],[-20,15],'Color','r','LineWidth', 2);
ylim([-80,80]);
title('item embedding of DICE on Movielens-10M dataset');

[~, objh] = legend('DICE-int', 'DICE-con','Fontsize', 16,'FontWeight','bold');
objhl = findobj(objh, 'type', 'text');
set(objhl, 'Fontsize', 16);
objhl = findobj(objh, 'type', 'patch');
set(objhl, 'Markersize', 12);
a = get(gca,'TickLabel');  
set(gca,'TickLabel',a,'fontsize',16,'FontWeight','bold');
box on;

%% Netflix item embeddings (Figure 5(b) right)
data = load('./data/dice_nf.txt');
pop = load('./data/nf_popularity.txt');
mypop = zeros(8331,1);
mypop(pop<10)=1;
mypop((pop>=17) & (pop<38))=2;
mypop(pop>38)=3;

figure();
scatter(data(1:8331,1), data(1:8331,2),40,mypop,'x','DisplayName', 'DICE-int');hold on;
scatter(data(8332:16662,1), data(8332:16662,2),40,mypop+3,'.','DisplayName', 'DICE-con');hold on;
plot([0,0],[-60,50],'Color','r','LineWidth', 2);
ylim([-80,60]);
title('item embedding of DICE on Netflix dataset');

[~, objh] = legend('DICE-int', 'DICE-con','Fontsize', 16,'FontWeight','bold');
objhl = findobj(objh, 'type', 'text');
set(objhl, 'Fontsize', 16);
objhl = findobj(objh, 'type', 'patch');
set(objhl, 'Markersize', 12);
a = get(gca,'TickLabel');  
set(gca,'TickLabel',a,'fontsize',16,'FontWeight','bold');
box on;

%% item embedding with conformity modeling (Figure 7 left)
data = load('./data/dice_pop.txt');
pop = load('./data/ml_popularity.txt');
mypop = zeros(4819,1);
mypop(pop<17)=1;
mypop((pop>=17) & (pop<71))=2;
mypop(pop>71)=3;

figure();
scatter(data(mypop==1,1), data(mypop==1,2),40,'r','.','DisplayName', 'unpopular');hold on;
scatter(data(mypop==2,1), data(mypop==2,2),40,'g','.','DisplayName', 'normal');hold on;
scatter(data(mypop==3,1), data(mypop==3,2),40,'b','.','DisplayName', 'popular');
[~, objh] = legend('unpopular', 'normal','popular','Fontsize', 16,'FontWeight','bold');
objhl = findobj(objh, 'type', 'text');
set(objhl, 'Fontsize', 16);
objhl = findobj(objh, 'type', 'patch');
set(objhl, 'Markersize', 12);
a = get(gca,'TickLabel');  
set(gca,'TickLabel',a,'fontsize',16,'FontWeight','bold');
title('DICE w/ conformity modeling');
box on;

%% item embedding without conformity modeling (Figure 7 right)
data = load('./data/dice_nopop.txt');
pop = load('./data/ml_popularity.txt');
mypop = zeros(4819,1);
mypop(pop<17)=1;
mypop((pop>=17) & (pop<71))=2;
mypop(pop>71)=3;

figure();
scatter(data(mypop==1,1), data(mypop==1,2),40,'r','.','DisplayName', 'unpopular');hold on;
scatter(data(mypop==2,1), data(mypop==2,2),40,'g','.','DisplayName', 'normal');hold on;
scatter(data(mypop==3,1), data(mypop==3,2),40,'b','.','DisplayName', 'popular');
[~, objh] = legend('unpopular', 'normal','popular','Fontsize', 16,'FontWeight','bold');
objhl = findobj(objh, 'type', 'text');
set(objhl, 'Fontsize', 16);
objhl = findobj(objh, 'type', 'patch');
set(objhl, 'Markersize', 12);
a = get(gca,'TickLabel');  
set(gca,'TickLabel',a,'fontsize',16,'FontWeight','bold');
title('DICE w/o popularity modeling');
box on;
