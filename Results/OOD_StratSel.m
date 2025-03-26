f1 = readtable('./Dataset/OOD_StratSwitch_Cyborg.csv');

Episode1 = table2array(f1(:,"Episode"));
Timestep1 = table2array(f1(:,"Step_no"));
%Freq1 = table2array(f1(:,"Freq"));

f2 = readtable('./Dataset/OOD_StratSwitch_SafeAction.csv');

Episode2 = table2array(f2(:,"Episode"));
Timestep2 = table2array(f2(:,"Step_no"));
%Freq2 = table2array(f2(:,"Freq"));

B1 = [];
count = 0;
for j=1:1000
    flag = 0;
    for i=1:size(Episode1)
        if Episode1(i,:) == j-1 
            if flag == 0
                count = 0;
                flag = 1;
            else
                count = count + 1;
            end
        end
    end
    B1(j) = count;
end

%disp(B1);

B2 = [];
count = 0;
for j=1:1000
    flag = 0;
    for i=1:size(Episode2) 
        if Episode2(i,:) == j-1 
            if flag == 0
                count = 0;
                flag = 1;
            else
                count = count + 1;
            end
        end
    end
    B2(j) = count;
end

T1 = ones(size(B1));
T2 = 2*ones(size(B2));

figure


Data = [B1; B2];
X = [T1; T2];
%disp(size(Data));
%disp(size(X));

axis = [0 3 -10 110];

Values = {'Strategy Switch (Normal)','Strategy Switch (Safe)'};

B = boxplot(Data(:),X(:),'Labels',Values,'Whisker',inf,'Notch',false);

ylabel('\bf{Number of OOD Transitions}','fontsize',30,'Interpreter','latex','FontWeight','bold');
xlabel('\bf{Strategy Switch (\boldmath$Meander \rightarrow B\_line$)}','fontsize',30,'FontWeight','bold','Interpreter','latex');
set(gca, 'XTick', [1, 2, 3, 4]); 
set(gca, 'XTickLabel', {'{\bf{No \boldmath$SAFE$ Switch}}','{\bf{\boldmath$SAFE$ Switch}}'},'TickLabelInterpreter','latex','fontweight','bold','fontsize',30);
set(gca, 'YTick', [0,10,20,30,40,50,60,70,80,90,100]); 
set(gca, 'YTickLabel', {'{\boldmath$0$}','{\boldmath$10$}','{\boldmath$20$}','{\boldmath$30$}','{\boldmath$40$}','{\boldmath$50$}','{\boldmath$60$}','{\boldmath$70$}','{\boldmath$80$}','{\boldmath$90$}','{\boldmath$100$}'},'TickLabelInterpreter','latex','fontweight','bold','fontsize',30);
title(['\bf{OOD Transitions with/without SAFE Switch}'],'FontSize',30,'FontWeight','bold','Interpreter','latex');

box_vars = findobj(gca,'Tag','Box');
set(findobj(gca,'Type','text'),'FontSize',30,'interpreter','latex','VerticalAlignment','middle') % to set Xaxis

set(B(:,1:2),'LineWidth',3.0);


