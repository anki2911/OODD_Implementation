import pandas as pd
import statistics

def recall(tp, tpfn ):
    return len(tp)/len(tpfn)

def precision_test(blue_act,blue_outcome,red_sessions):
    def y_determine_red_access(session_list):
        #print('-> In pt, session list is:',session_list)
        
        #print('-> session list agent:',session_list['Agent'])

        # Process only the keys that exist in the dictionary
        if session_list['Agent'] == 'Red':
          if "Username" in session_list:
            privileged = session_list['Username'] in {'root','SYSTEM'}
            if privileged:
               return 'Root'
          else: return 'User'
        else: return 'None'


    ret = None
    print('-> In pt, Blue action is:',blue_act)
    _act=str(blue_act).split(' ')[0]
    _target= str(blue_act).split(' ')[1]
   
    
    #print('->In precision test, last action is:',_act)
    if (_act=='Restore') or (_act=='Remove'):
        #print('In precision test, Target isL',_target)
        #print('-> in Pt , red session is:',red_sessions)
        red_foothold= 'None'
        
        if _target in red_sessions:  
          print('red sessions is :',red_sessions)
          true_state= red_sessions[_target]
          print('->In pt true state:',true_state[0][0], 'its type is:',type(true_state[0][0]))
          print('->In pt blue outcome is:',blue_outcome)
          red_foothold = y_determine_red_access(true_state[0][0])
          print('-> red foothold in test precison:',red_foothold)

        if red_foothold == 'None':
            ret = 'FP'
        elif red_foothold == 'User':
            ret = 'FP' if _act == 'Restore' else 'TP' # Positive reward if actually needed removal
        elif red_foothold == 'Root':
            ret = 'TP' if _act == 'Restore' else 'FP' # Just wasting time to remove rooted machine
        ret = (_act, ret)
    return ret


def precision(env, last_act,blue_outcome):
    def determine_red_access(session_list):
        '''
        Stolen from TrueTableWrapper
        '''
        print('From precision, session list is:',session_list)
        for session in session_list:
            if session['Agent'] != 'Red':
                continue
            privileged = session['Username'] in {'root','SYSTEM'}
            return 'Root' if privileged else 'User'

        return 'None'

    ret = None
    print('last action is:',last_act)
    _act=str(last_act).split(' ')[0]
    _target= str(last_act).split(' ')[1]
    last_act=_act
    target= _target
    print('last action is:',last_act)
    if (last_act=='Restore') or (last_act=='Remove'):
        print('Target isL',target)
        true_state = env.get_agent_state('True')[target]['Sessions']
        print('-> true state:',true_state)
        print('-> blue outcome is:',blue_outcome)
        red_foothold = determine_red_access(true_state)
        print('-> red foothold:',red_foothold)
        last_act_str = str(last_act).split(' ')[0]

        if red_foothold == 'None':
            ret = 'FP'
        elif red_foothold == 'User':
            ret = 'FP' if last_act_str == 'Restore' else 'TP' # Positive reward if actually needed removal
        elif red_foothold == 'Root':
            ret = 'TP' if last_act_str == 'Restore' else 'FP' # Just wasting time to remove rooted machine

        ret = (last_act_str, ret)

    return ret


def availability_discrete(data_file_loc):
  # Create a DataFrame
  
  df = pd.read_csv(data_file_loc)
  #row_count = len(df)
  # Initialize counters or flags
  row_count = []
  impacted_steps= 0
  remove_restore_success = 0
  impacted=0
  impacted_steps_count_array=[]

  # Iterate through the DataFrame
  for index, row in df.iterrows():
    row_count.append(index+1)
    # Check if Red Action is "Impact Op_Server0" and Red success is True
    if row['Red Action'] == "Impact Op_Server0" and row['Red success']:
        impacted=1
   
    if impacted==1:
        impacted_steps+=1
    impacted_steps_count_array.append(impacted_steps)
    
    # Check if Blue Action is "Remove" or "Restore" on "Op_Server0" and Blue success is True
    if ("Remove" in row['Blue Action'] or "Restore" in row['Blue Action']) and "Op_Server0" in row['Blue Action'] and row['Blue success']:
        remove_restore_success += 1
        impacted=0
    
  # Print the results
  print(f"Number of successful 'Impact Op_Server0' actions: {impacted_steps}")
  print(f"Number of successful 'Remove' or 'Restore' actions on 'Op_Server0': {remove_restore_success}")
  return impacted_steps_count_array,row_count



import matplotlib.pyplot as plt

def plot_avl(list1, list2, list3, list4):
    # Create a range for the x-axis
    print('Plotting...')
    x = range(len(list1))

    # Plot each list with enhanced styling
    plt.plot(x, list1, label="Episode 1", color="green", linewidth=2, marker='o',markersize=12)  # No agent with markers
    plt.plot(x, list2, label="Episode 2", color="red", linewidth=2, marker='s',markersize=7)  # Only red with dashed line
    plt.plot(x, list3, label="Episode 3", color="blue", linewidth=2, marker='^',markersize=4)  # Both with dash-dot line
    plt.plot(x, list4, label="Episode 4", color="black", linewidth=2, marker='+',markersize=2)  # Both with dash-dot line

    # Add title and axis labels with improved text
    plt.title(
        "Comparison of Availability in four episodes over 50 steps \n with B_lineAgent as red agent and EBT as blue agent",
        fontsize=14, fontweight='bold'
    )
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Availability fraction", fontsize=12)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a legend with improved styling
    plt.legend(loc="best", fontsize=10)

    # Save the plot as a high-resolution file
    plt.savefig("comparison_plot.png", dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

    return

def plot_recall(list1, list2, list3, list4, plot_type):
    # Create a range for the x-axis
    print('Plotting...')
    
    Data = {'EBT': list1, 'Dartmouth': list2, 'Cardiff': list3, 'Cornell': list4}
    
    x = [[1], [2], [3], [4]]
    color_label = ["green", "blue", "red", "black"]

    ax = plt.axes()

    i = 0
    for ID in Data.keys():
        plt.boxplot(Data[ID], positions=x[i], patch_artist=True, boxprops=dict(facecolor=color_label[i]), notch=False)
        i = i + 1



    ax.set_xticklabels(['EBT', 'Dartmouth','Cardiff', 'Cornell'], fontsize = 14)
    
    # Add title and axis labels with improved text
    if str(plot_type) == "recall":
        plt.title(
        "Recall values for different agents over 4 episodes",
        fontsize=14, fontweight='bold'
        )
        plt.xlabel("Agents", fontsize=12)
        plt.ylabel("Recall values", fontsize=12)
        # Save the plot as a high-resolution file
        plt.savefig("recall_plot.png", dpi=300, bbox_inches='tight')
    else:
        plt.title(
        "Precision values for different agents over 4 episodes",
        fontsize=14, fontweight='bold'
        )
        plt.xlabel("Agents", fontsize=12)
        plt.ylabel("Precision values", fontsize=12)
        # Save the plot as a high-resolution file
        plt.savefig("precision_plot.png", dpi=300, bbox_inches='tight')

    ## Add a grid for better readability
    #plt.grid(True, linestyle='--', alpha=0.7)

    

    # Display the plot
    plt.show()

    return


def extract_precision_recall(data_file):
    with open(data_file, "r") as f:
        for lines in f:
            if lines.startswith("Recall metrics is: "):
                string = lines.strip("\n")
                string = string.split(" ")
                recall = float(string[-1])
            if lines.startswith("precision metrics : "):
                string = lines.strip("\n")
                string = string.split(" ")
                num_tp = 0
                num_fp = 0
                for words in string:
                    if words.startswith("'TP"):
                        num_tp += 1
                    if words.startswith("'FP"):
                        num_fp += 1
            
    return recall,num_tp,num_fp




if __name__ == "__main__":
    #data= './logs/20250108154311_BT_dart_ne_RT_blineagent_emu_log_success.csv'
    #data1='./logs/20250108142345_BT_sleep_RT_sleep_emu_log_success.csv'
    #data2='./logs/20250108142541_BT_sleep_RT_blineagent_emu_log_success.csv'
    
    
    #data3= './logs/20250108154311_BT_dart_ne_RT_blineagent_emu_log_success.csv'
    #data3= './logs/20250109083921_BT_cornell_RT_blineagent_emu_log_success.csv'
    data1 = './logs/EBT_50_emu_log_success_1.csv'
    data2 = './logs/EBT_50_emu_log_success_2.csv'
    data3 = './logs/EBT_50_emu_log_success_3.csv'
    data4 = './logs/EBT_50_emu_log_success_4.csv'


    impacted1,row_count=availability_discrete(data1)
    print(impacted1)
    print(row_count)
    avl_result1 = [(1-(impacted1[x]*0.1/row_count[x]))  for x in range(len(impacted1))]
    print(avl_result1)

    impacted2,row_count=availability_discrete(data2)
    avl_result2 = [(1-(impacted2[x]*0.1/row_count[x]))  for x in range(len(impacted2))]
    print(avl_result2)

    impacted3,row_count=availability_discrete(data3)
    avl_result3 = [(1-(impacted3[x]*0.1/row_count[x]))  for x in range(len(impacted3))]
    print(avl_result3)
    
    
    impacted4,row_count=availability_discrete(data4)
    avl_result4 = [(1-(impacted4[x]*0.1/row_count[x]))  for x in range(len(impacted4))]
    print(avl_result4)
    plot_avl(avl_result1,avl_result2,avl_result3,avl_result4)


    Recall_1 = []
    Precision_1 = []

    pre_data1 = './logs/EBT_50_emu_action_obs_1.txt'
    recall1, num_tp1, num_fp1 = extract_precision_recall(pre_data1)
    Recall_1.append(recall1)
    Precision_1.append(float(num_tp1/(num_tp1+num_fp1)))

    pre_data2 = './logs/EBT_50_emu_action_obs_2.txt'
    recall2, num_tp2, num_fp2 = extract_precision_recall(pre_data2)
    Recall_1.append(recall2)
    Precision_1.append(float(num_tp2/(num_tp2+num_fp2)))
    
    pre_data3 = './logs/EBT_50_emu_action_obs_3.txt'
    recall3, num_tp3, num_fp3 = extract_precision_recall(pre_data3)
    Recall_1.append(recall3)
    Precision_1.append(float(num_tp3/(num_tp3+num_fp3)))
    
    pre_data4 = './logs/EBT_50_emu_action_obs_4.txt'
    recall4, num_tp4, num_fp4 = extract_precision_recall(pre_data4)
    Recall_1.append(recall4)
    Precision_1.append(float(num_tp4/(num_tp4+num_fp4)))

    mean_recall = statistics.mean(Recall_1)
    std_dev = statistics.stdev(Recall_1)

    print(mean_recall)
    print(std_dev)

    plot_recall(Recall_1, Recall_1, Recall_1, Recall_1,"recall")
    plot_recall(Precision_1,Precision_1,Precision_1,Precision_1,"precision")
    
    

    

    