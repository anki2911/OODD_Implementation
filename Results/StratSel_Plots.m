f = readtable('./Dataset/OOD_StratSwitch_Cyborg.csv');
g = readtable('./Dataset/OOD_StratSwitch_SafeAction.csv');


Episode1 = table2array(f(:,"Episode"));
Timestep1 = table2array(f(:,"Step_no"));
%Reward1 = table2array(f(:,"Reward"));


Episode2 = table2array(g(:,"Episode"));
Timestep2 = table2array(g(:,"Step_no"));
%Reward2 = table2array(g(:,"Reward"));

episodes_wc = [1,2,42,64,91];
    
episodes_sa = [1,2,73,33,116];


Episode3 = table2array(f1(:,"Episode"));
Timestep3 = table2array(f1(:,"Step_no"));

Episode4 = table2array(g1(:,"Episode"));
Timestep4 = table2array(g1(:,"Step_no"));

OOD_Cyborg_0 = [];
ind10 = 1;
OOD_Cyborg_1 = [];
ind11 = 1;
OOD_Cyborg_2 = [];
ind12 = 1;
OOD_Cyborg_3 = [];
ind13 = 1;
OOD_Cyborg_4 = [];
ind14 = 1;

OOD_SA_0 = [];
ind20 = 1;
OOD_SA_1 = [];
ind21 = 1;
OOD_SA_2 = [];
ind22 = 1;
OOD_SA_3 = [];
ind23 = 1;
OOD_SA_4 = [];
ind24 = 1;

for i=1:5
    ep_no = episodes_wc(i);
    for j=1:size(Episode1)
        if Episode1(j) == ep_no 
            if i == 1
                OOD_Cyborg_0(ind10,:) = Timestep1(j);
                ind10 = ind10 + 1;
            elseif i == 2
                OOD_Cyborg_1(ind11,:) = Timestep1(j);
                ind11 = ind11 + 1;
            elseif i == 3
                OOD_Cyborg_2(ind12,:) = Timestep1(j);
                ind12 = ind12 + 1;
            elseif i == 4
                OOD_Cyborg_3(ind13,:) = Timestep1(j);
                ind13 = ind13 + 1;
            elseif i == 5
                OOD_Cyborg_4(ind14,:) = Timestep1(j);
                ind14 = ind14 + 1;
            end
        end
    end
end 

for i=1:5
    ep_no = episodes_sa(i);
    for j=1:size(Episode2)
        if Episode2(j) == ep_no 
            if i == 1
                OOD_SA_0(ind20,:) = Timestep2(j);
                ind20 = ind20 + 1;
            elseif i == 2
                OOD_SA_1(ind21,:) = Timestep2(j);
                ind21 = ind21 + 1;
            elseif i == 3
                OOD_SA_2(ind22,:) = Timestep2(j);
                ind22 = ind22 + 1;
            elseif i == 4
                OOD_SA_3(ind23,:) = Timestep2(j);
                ind23 = ind23 + 1;
            elseif i == 5
                OOD_SA_4(ind24,:) = Timestep2(j);
                ind24 = ind24 + 1;
            end
        end
    end
end 

Switch_1 = [];
for i=1:5
    ep_no = episodes_wc(i);
    for j=1:size(Episode3)
        if Episode3(j,:) == ep_no
            Switch_1(i) = Timestep3(j);
        end
    end
end

Switch_2 = [];
for i=1:5
    ep_no = episodes_sa(i);
    for j=1:size(Episode4)
        if Episode4(j,:) == ep_no
            Switch_2(i) = Timestep4(j);
        end
    end
end

T = [];
for i=1:100
    T(i) = i;
end

disp(Switch_1);
disp(Switch_2);

figure;
for i = 1:5
    for j=1:100
        if i == 1
            if OOD_Cyborg_0(1) == j
                l = size(OOD_Cyborg_0);
                for k=1:j-1
                    OOD_Cyborg_0(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_Cyborg_0(k) = 1;
                end
                plot(T(:),OOD_Cyborg_0(:),'x',color=[1 0 0],MarkerSize=15,LineWidth=2);
                hold on;
                j = 200;
            end
        end
        if i == 2
            if OOD_Cyborg_1(1) == j
                l = size(OOD_Cyborg_1);
                for k=1:j-1
                    OOD_Cyborg_1(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_Cyborg_1(k) = 1;
                end
                plot(T(:),OOD_Cyborg_1(:),'x',color=[0 0 1],MarkerSize=15,LineWidth=2);
                hold on;
                j = 200;
            end
        end
        if i == 3
            if OOD_Cyborg_2(1) == j
                l = size(OOD_Cyborg_2);
                for k=1:j-1
                    OOD_Cyborg_2(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_Cyborg_2(k) = 1;
                end
                plot(T(:),OOD_Cyborg_2(:),'x',color=[0 0 0],MarkerSize=15,LineWidth=2);
                hold on;
                j = 200;
            end
        end
        if i == 4
            if OOD_Cyborg_3(1) == j
                l = size(OOD_Cyborg_3);
                for k=1:j-1
                    OOD_Cyborg_3(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_Cyborg_3(k) = 1;
                end
                plot(T(:),OOD_Cyborg_3(:),'x',color=[0 0.5 0],MarkerSize=15,LineWidth=2);
                hold on;
                j = 200;
            end
        end
        if i == 5
            if OOD_Cyborg_4(1) == j
                l = size(OOD_Cyborg_4);
                for k=1:j-1
                    OOD_Cyborg_4(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_Cyborg_4(k) = 1;
                end
                plot(T(:),OOD_Cyborg_4(:),'x',color=[0.58 0.29 0],MarkerSize=15,LineWidth=2);
                hold on;
                j = 200;
            end
        end
    end
end

for i = 1:5
    for j=1:100
        if i == 1
            if OOD_SA_0(1) == j
                l = size(OOD_SA_0);
                for k=1:j-1
                    OOD_SA_0(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_SA_0(k) = 1;
                end
                for k=j+l:100
                    OOD_SA_0(k) = 0.1;
                end
                plot(T(:),OOD_SA_0(:),'--',color=[1 0 0],MarkerSize=20,LineWidth=4);
                hold on;
                j = 200;
            end
        end
        if i == 2
            if OOD_SA_1(1) == j
                l = size(OOD_SA_1);
                for k=1:j-1
                    OOD_SA_1(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_SA_1(k) = 1;
                end
                for k=j+l:100
                    OOD_SA_1(k) = 0.1;
                end
                plot(T(:),OOD_SA_1(:),'--',color=[0 0 1],MarkerSize=20,LineWidth=4);
                hold on;
                j = 200;
            end
        end
        if i == 3
            if OOD_SA_2(1) == j
                l = size(OOD_SA_2);
                for k=1:j-1
                    OOD_SA_2(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_SA_2(k) = 1;
                end
                for k=j+l:100
                    OOD_SA_2(k) = 0.1;
                end
                plot(T(:),OOD_SA_2(:),'--',color=[0 0 0],MarkerSize=20,LineWidth=4);
                hold on;
                j = 200;
            end
        end
        if i == 4
            if OOD_SA_3(1) == j
                l = size(OOD_SA_3);
                for k=1:j-1
                    OOD_SA_3(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_SA_3(k) = 1;
                end
                for k=j+l:100
                    OOD_SA_3(k) = 0.1;
                end
                plot(T(:),OOD_SA_3(:),'--',color=[0 0.5 0],MarkerSize=20,LineWidth=4);
                hold on;
                j = 200;
            end
        end
        if i == 5
            if OOD_SA_4(1) == j
                l = size(OOD_SA_4);
                for k=1:j-1
                    OOD_SA_4(k) = 0.1;
                end
                for k=j:j+l-1
                    OOD_SA_4(k) = 1;
                end
                for k=j+l:100
                    OOD_SA_4(k) = 0.1;
                end
                plot(T(:),OOD_SA_4(:),'--',color=[0.58 0.29 0],MarkerSize=20,LineWidth=4);
                hold on;
                j = 200;
            end
        end
    end
end

h = zeros(10, 1);
h(1) = plot(NaN,NaN,'x','color',[1 0 0],'MarkerSize',20,'Linewidth',3.0);
h(2) = plot(NaN,NaN,'x','color',[0 0 0],'MarkerSize',20,'Linewidth',3.0);
h(3) = plot(NaN,NaN,'x','color',[0 0.5 0],'MarkerSize',20,'Linewidth',3.0);
h(4) = plot(NaN,NaN,'x','color',[0 0 1],'MarkerSize',20,'Linewidth',3.0);
h(5) = plot(NaN,NaN,'x','color',[0.58 0.29 0],'MarkerSize',20,'Linewidth',3.0);

h(6) = plot(NaN,NaN,'--','color',[1 0 0],'MarkerSize',20,'Linewidth',3.0);
h(7) = plot(NaN,NaN,'--','color',[0 0 0],'MarkerSize',20,'Linewidth',3.0);
h(8) = plot(NaN,NaN,'--','color',[0 0.5 0],'MarkerSize',20,'Linewidth',3.0);
h(9) = plot(NaN,NaN,'--','color',[0 0 1],'MarkerSize',20,'Linewidth',3.0);
h(10) = plot(NaN,NaN,'--','color',[0.58 0.29 0],'MarkerSize',20,'Linewidth',3.0);


%axis([0 150 0 1.2]);
legend(h,{'Episode 1','Episode 2','Episode 3','Episode 4','Episode 5','Episode 1 (SAFE)','Episode 2 (SAFE)','Episode 3 (SAFE)','Episode 4 (SAFE)','Episode 5 (SAFE)'},'Location','northeast','fontsize',20);
xticks([0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150]);
yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5]);
xticklabels({'{\boldmath$0$}','{\boldmath$10$}','{\boldmath$20$}','{\boldmath$30$}','{\boldmath$40$}','{\boldmath$50$}','{\boldmath$60$}','{\boldmath$70$}','{\boldmath$80$}','{\boldmath$90$}','{\boldmath$100$}','{\boldmath$110$}','{\boldmath$120$}','{\boldmath$130$}','{\boldmath$140$} ','{\boldmath$150$}'});
yticklabels({'','{\boldmath$ID$}','','','','','','','','','{\boldmath$OOD$}',' ',' ',' ',' ',' '});
set(gca,'TickLabelInterpreter','latex');
set(gca,'fontweight','bold','fontsize',30);     
xlabel('\bf{Timesteps}','fontsize',30,'interpreter','latex')
ylabel('\bf{Runtime Behavior}','fontsize',30,'interpreter','latex')
hold off;
