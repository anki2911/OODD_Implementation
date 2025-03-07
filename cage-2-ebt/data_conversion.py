import sys
import csv

def main():
    action = {}
    count = 0
    with open('BlueSpace.csv', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for lines in reader:
            if count > 0:
                action[str(lines[0])] = int(lines[1])
            count += 1
    #print(action)
    count = 0
    f = open("./emu_data/D_bline_50_3.txt","w")
    g = open("./emu_data/A_bline_50_3.txt","w")
    h = open("./emu_data/R_bline_50_3.txt","w")

    with open('./logs/EBT_50_emu_data_4.csv', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for lines in reader:
            if count == 0:
                for i in range(52):
                    f.write(str(0))
                    f.write(" ")
                f.write("\n")
            if count > 0:
                val = action[str(lines[1])]
                g.write(str(val))
                g.write("\n")
                string = lines[3]
                string = string.strip("[")
                string = string.strip("]")
                for c in string:
                    if c == '0' or c == '1':
                        f.write(c)
                        f.write(" ")
                f.write("\n")
                h.write(str(lines[4]))
                h.write("\n")

            count += 1
    
     

if __name__ == '__main__':
    main()