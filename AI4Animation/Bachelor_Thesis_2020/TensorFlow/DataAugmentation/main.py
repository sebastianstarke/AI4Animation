import csv
import pandas as pd
import numpy as np
def main():

    inputOriginal = pd.read_csv('Input_Original.txt', sep=" ", header=None)
    zeroCol = [0] * len(inputOriginal.index)
    labels = pd.read_csv('InputLabels_Original.txt', sep=" ", header=None)
    inputOriginal.columns = labels[1]

    startStyle = 7
    startCol = 12
    for x in range (1,4):
        loc = startCol + x

        for i in range(1,13):
            location = (loc*i)+i-1
            col = "Trajectory"+str(i)+"Style"+str(startStyle)
            print(col)
            print("loc: " + str(location))
            inputOriginal.insert(loc=location, column=col, value=zeroCol)

        startStyle+=1


    inputNew = pd.read_csv('Input.txt', sep=" ", header=None)
    labels_inputNew = pd.read_csv('InputLabels.txt', sep=" ", header=None)
    inputNew.columns = labels_inputNew[1]

    df = inputOriginal.append(inputNew)

    df.to_csv("Input_Final.txt", float_format='%.5f', header=None, index=None,  sep=" ", mode='a')
    print(len(df.columns))

    outputOriginal = pd.read_csv('Output_Original.txt', sep=" ", header=None)
    outputNew = pd.read_csv('Output.txt', sep=" ", header=None)
    dfOutput = outputOriginal.append(outputNew)
    dfOutput.to_csv("Output_Final.txt", float_format='%.5f', header=None, index=None, sep=" ", mode='a')

    print(len(dfOutput.columns))

def breakdown():

    data = pd.read_csv('Input_Final.txt', sep=" ", header=None)
    labels_inputNew = pd.read_csv('InputLabels.txt', sep=" ", header=None)

    data.columns = labels_inputNew[1]
    rowCount = len(data.index)

    styleDF = data[['Trajectory7Style1', 'Trajectory7Style2','Trajectory7Style3','Trajectory7Style4',
            'Trajectory7Style5','Trajectory7Style6','Trajectory7Style7','Trajectory7Style8','Trajectory7Style9']].copy()
    styleDF['Max'] = styleDF.idxmax(axis=1) #return max column of the dataframe e.g. Trajectory7Style1
    frames = styleDF.groupby('Max').count().iloc[:,0]

    sum = 0
    for x in frames:
        percentage = (x / rowCount) * 100
        print("%.2f & %.0f & %.2f \\\\" % ((x/60) , x , percentage))
        sum += percentage

def contactLabels():
    data = pd.read_csv('Output_Final.txt', sep=" ", header=None)
    outputLabels = pd.read_csv('OutputLabels.txt', sep=" ", header=None)
    data.columns = outputLabels[1]

    #Bone 11, 16, 20, 24
    bone = 11
    for i in range (1,5):
        posY = data["Bone"+str(bone)+"PositionY"]
        magnitude = np.linalg.norm(data[['Bone'+str(bone)+'VelocityX','Bone'+str(bone)+'VelocityY','Bone'+str(bone)+'VelocityZ']].values,axis=1)

        result = []
        for j in range(posY.size):
            result.append(1.00000 if (posY[j] < 0.025 and magnitude[j] < 1) else 0.00000)

        data["Bone"+str(bone)+"Contact"] = result
        print(data)
        if(i==1):
            bone+=5
        else:
            bone+=4

    data.to_csv("Output_Final_Contact.txt", float_format='%.5f', header=None, index=None, sep=" ", mode='a')
    print(len(data.columns))

if __name__ =='__main__':
    main()
    contactLabels()
    breakdown()
