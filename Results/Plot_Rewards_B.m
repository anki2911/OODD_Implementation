f = readtable('./Dataset/Dist_Prob_B_100.csv');

Episode = table2array(f(1:97000,"Episode"));
Timestep = table2array(f(1:97000,"Step_no"));
Reward = table2array(f(1:97000,"Reward"));
Freq = table2array(f(1:97000,"Freq"));

B1 = [];
x1 = [];
i1 = 1;
B2 = [];
x2 = [];
i2 = 1;
B3 = [];
x3 = [];
i3 = 1;
B4 = [];
x4 = [];
i4 = 1;
B5 = [];
x5 = [];
i5 = 1;
B6 = [];
x6 = [];
i6 = 1;

for i=1:97000
    pdf = log10(Freq(i,:)/97000);
    %disp(pdf);
    if Freq(i,:) == 0
        B1(i1,:) = Reward(i,:);
        x1(i1,:) = 1;
        i1 = i1 + 1;
    %elseif pdf > -6 && pdf <= -5
    %    disp("Hello");
    %    B2(i2,:) = Reward(i,:);
    %    x2(i2,:) = 2;
    %    i2 = i2 + 1;
    elseif pdf <= -4
        B3(i3,:) = Reward(i,:);
        x3(i3,:) = 2;
        i3 = i3 + 1;
    elseif pdf <= -3
        B4(i4,:) = Reward(i,:);
        x4(i4,:) = 3;
        i4 = i4 + 1;
    elseif pdf <= -2
        B5(i5,:) = Reward(i,:);
        x5(i5,:) = 4;
        i5 = i5 + 1;
    elseif pdf <= -1
        B6(i6,:) = Reward(i,:);
        x6(i6,:) = 5;
        i6 = i6 + 1;
    end
end


figure


Data = [B1; B3; B4; B5];
X = [x1; x3; x4; x5];


Values = {'0','10^{-5}','10^{-4}','10^{-3}'};

B = boxplot(Data,X,'Labels',Values,'Whisker',inf,'Notch',false);

ylabel('\bf{Reward}','fontsize',20,'Interpreter','latex','FontWeight','bold');
xlabel('\bf{Transition Probability Threshold (\boldmath$\rho$)}','fontsize',20,'FontWeight','bold','Interpreter','latex');
set(gca, 'XTick', [1, 2, 3, 4]); 
set(gca, 'XTickLabel', {'{\boldmath$0$}','{\boldmath$10^{-5}$}', '{\boldmath$10^{-4}$}','{\boldmath$10^{-3}$}'},'TickLabelInterpreter','latex','fontweight','bold','fontsize',30);
set(gca, 'YTick', [-2,-1.5, -1,-0.5, 0]); 
set(gca, 'YTickLabel', {'{\boldmath$-2$}','{\boldmath$-1.5$}','{\boldmath$-1$}','{\boldmath$-0.5$}','{\boldmath$0$}'},'TickLabelInterpreter','latex','fontweight','bold','fontsize',30);
title(['\bf{Reward Distribution against \boldmath$B\_line$ strategy}'],'FontSize',30,'FontWeight','bold','Interpreter','latex');

box_vars = findobj(gca,'Tag','Box');
set(findobj(gca,'Type','text'),'FontSize',20,'interpreter','latex','VerticalAlignment','middle') % to set Xaxis

set(B(:,1:4),'LineWidth',3.0);


